"""Actual fills and explicit denominators must not invent returns or double fees."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.evaluation import PortfolioAnalysis


def test_supplied_initial_nav_survives_json_without_filling_later_missing_account_values():
    from ml4t.diagnostic.evaluation.portfolio_analysis import PortfolioInputData

    dates = pd.date_range("2025-07-07", periods=3, tz="UTC")
    analysis = PortfolioAnalysis.from_frames(
        pd.Series([-0.001, 0.01, 0.01], index=dates),
        valuations=pd.DataFrame({"date": dates, "equity": [999.0, None, 1019.0]}),
        transactions=pd.DataFrame(
            {
                "timestamp": dates,
                "asset": ["A"] * 3,
                "quantity": [1.0] * 3,
                "price": [100.0] * 3,
                "commission": [1.0] * 3,
                "slippage": [0.0] * 3,
            }
        ),
        calendar=None,
        initial_nav=1000,
    )
    timeline = analysis.compute_transaction_analysis().timeline
    assert timeline["beginning_nav"].to_list() == [1000.0, 999.0, None]
    assert timeline["observed_cost_return"].to_list() == [0.001, 1 / 999, None]
    restored = PortfolioInputData.from_json(analysis.input_data.to_json())
    assert restored.metadata["initial_nav"] == 1000


@pytest.mark.parametrize("initial_nav", [-1.0, float("nan"), float("inf"), True])
def test_invalid_initial_nav_is_rejected_before_account_analysis(initial_nav):
    returns = pd.Series([0.01], index=pd.date_range("2025-07-07", periods=1))
    with pytest.raises(ValueError, match="initial_nav"):
        PortfolioAnalysis.from_frames(returns, initial_nav=initial_nav)
    assert returns.iloc[0] == 0.01


def _account(*, unknown_cost=False, missing_nav=False, calendar="CME_Equity"):
    dates = pd.date_range("2025-07-07", periods=3, tz="UTC")
    returns = pd.Series([0.01, -0.02, 0.03], index=dates)
    valuations = pd.DataFrame(
        {
            "date": pd.DatetimeIndex(["2025-07-03T00:00Z", *dates]),
            "equity": [1000.0, None if missing_nav else 1100.0, 1200.0, 1300.0],
        }
    )
    positions = pd.DataFrame({"A": [200.0, 400.0, 600.0]}, index=dates)
    fills = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-07-06T23:30Z", "2025-07-07T00:30Z"]),
            "asset": ["A", "A"],
            "quantity": [0.5, -0.5],
            "price": [2.0, 2.0],
            "multiplier": [100.0, 100.0],
            "commission": [None if unknown_cost else 0.1, 0.1],
            "slippage": [0.0, 0.0],
        }
    )
    return PortfolioAnalysis.from_frames(
        returns,
        positions=positions,
        valuations=valuations,
        transactions=fills,
        complete_snapshots=True,
        feed_spec={
            "timestamp_col": "date",
            "entity_col": "asset",
            "time_contract_version": 2,
            "timestamp_semantics": "session_label",
            "calendar": calendar,
        },
        transaction_timestamp_col="timestamp",
    )


def test_overnight_fills_share_one_exchange_session_and_use_beginning_not_closing_nav():
    analysis = _account()
    result = analysis.compute_transaction_analysis(incremental_bps=[0, 10])
    assert result.executions["session"].to_list() == [pd.Timestamp("2025-07-07").date()] * 2
    assert result.executions["filled_notional"].to_list() == [100.0, 100.0]
    assert result.timeline["filled_notional"].to_list() == [200.0, 0.0, 0.0]
    assert result.timeline["denominator"].to_list() == [1000.0, 1100.0, 1200.0]
    assert result.timeline["turnover"].to_list() == pytest.approx([0.2, 0.0, 0.0])
    assert result.metrics["mean_turnover_all"] == pytest.approx(0.2 / 3)
    assert result.metrics["mean_turnover_active"] == pytest.approx(0.2)
    scenario = result.scenarios.filter(pl.col("incremental_bps") == 10)
    assert scenario["incremental_cost"].to_list() == pytest.approx([0.2, 0, 0])
    assert scenario["incremental_cost_return"].to_list() == pytest.approx([0.0002, 0, 0])
    assert scenario["net_return"].to_list() == pytest.approx([0.0098, -0.02, 0.03])
    np.testing.assert_array_equal(analysis.returns, [0.01, -0.02, 0.03])
    assert result.scenarios.filter(pl.col("incremental_bps") == 0)["net_return"].to_list() == [
        0.01,
        -0.02,
        0.03,
    ]


def test_overnight_execution_timing_keeps_all_hours_and_converts_only_the_display_zone():
    result = _account().compute_transaction_analysis()
    assert result.timeline["absolute_units"].to_list() == [1.0, 0.0, 0.0]
    assert result.metrics["total_absolute_units"] == 1.0
    before = result.executions.clone()
    original = result.execution_timing()
    assert original.height == 288
    active = original.filter(pl.col("num_fills") > 0)
    assert active["time"].to_list() == ["00:30", "23:30"]
    assert active["absolute_units"].to_list() == [0.5, 0.5]
    assert active["filled_notional"].to_list() == [100.0, 100.0]
    assert active["notional_fraction"].to_list() == [0.5, 0.5]
    display = result.execution_timing(bin_minutes=60, timezone="America/New_York")
    assert display.height == 24
    assert display.filter(pl.col("num_fills") > 0)["time"].to_list() == ["19:00", "20:00"]
    assert result.executions.equals(before)


@pytest.mark.parametrize("bin_minutes", [0, 1441, True, 1.5])
def test_execution_timing_rejects_invalid_bin_widths_without_changing_results(bin_minutes):
    result = _account().compute_transaction_analysis()
    before = result.executions.clone()
    with pytest.raises((ValueError, TypeError), match="bin_minutes"):
        result.execution_timing(bin_minutes=bin_minutes)
    assert result.executions.equals(before)


def test_unknown_naive_execution_timezone_disables_only_timing_not_actual_activity():
    dates = pd.date_range("2025-07-07T15:00", periods=2)
    analysis = PortfolioAnalysis.from_frames(
        pd.Series([0.01, 0.02], index=dates),
        transactions=pd.DataFrame(
            {"timestamp": dates, "asset": [7, 7], "quantity": [1, -1], "price": [100, 110]}
        ),
        calendar=None,
        transaction_timestamp_col="timestamp",
    )
    result = analysis.compute_transaction_analysis()
    assert result.metrics["total_filled_notional"] == 210
    with pytest.raises(ValueError, match="declared feed timezone"):
        result.execution_timing()
    pytest.importorskip("plotly")
    report = analysis.create_tear_sheet(transactions=result)
    assert report.availability["execution_timing"]["status"] == "unavailable"
    assert "Execution Time Distribution" not in report.figures
    assert list(report.figures["Absolute Executed Units"].data[0].y) == [1, 1]


def test_unknown_observed_commission_never_becomes_zero_but_incremental_cost_is_known():
    result = _account(unknown_cost=True).compute_transaction_analysis(incremental_bps=[10])
    assert result.executions["implementation_cost"].null_count() == 1
    assert result.timeline["implementation_cost"].to_list() == [None, 0.0, 0.0]
    assert result.metrics["total_implementation_cost"] is None
    assert result.scenarios["net_return"][0] == pytest.approx(0.0098)


def test_missing_beginning_nav_keeps_notional_and_incremental_dollars_but_not_ratios():
    result = _account(missing_nav=True).compute_transaction_analysis(incremental_bps=[10])
    assert result.timeline["denominator"].to_list() == [1000.0, None, 1200.0]
    assert result.timeline["turnover"].to_list() == [0.2, None, 0.0]
    assert result.scenarios["incremental_cost"].to_list() == pytest.approx([0.2, 0, 0])
    assert result.scenarios["net_return"].to_list() == pytest.approx([0.0098, -0.02, 0.03])


def test_average_gross_book_and_one_way_turnover_are_explicit_distinct_methods():
    result = _account().compute_transaction_analysis(
        turnover_denom="average_gross_book", turnover_method="one_way", incremental_bps=[10]
    )
    assert result.timeline["denominator"].to_list() == [None, 300.0, 500.0]
    assert result.timeline["turnover"].to_list() == [None, 0.0, 0.0]
    # Hypothetical cost return always uses NAV, never the turnover exposure denominator.
    assert result.scenarios["incremental_cost_return"][0] == pytest.approx(0.0002)


def test_returns_only_account_does_not_claim_an_empty_execution_ledger():
    analysis = PortfolioAnalysis.from_frames(
        pd.Series([0.01], index=pd.date_range("2025-07-07", periods=1)), calendar=None
    )
    with pytest.raises(ValueError, match="transactions.*not supplied"):
        analysis.compute_transaction_analysis()


@pytest.mark.parametrize("bps", [[True], [-1], [np.nan], [np.inf], ["10"]])
def test_invalid_cost_scenarios_leave_owned_inputs_unchanged(bps):
    analysis = _account()
    before = analysis.input_data.to_json()
    with pytest.raises((ValueError, TypeError), match="bps"):
        analysis.compute_transaction_analysis(incremental_bps=bps)
    assert analysis.input_data.to_json() == before


def test_explicit_none_matches_exact_nanosecond_rows_without_session_grouping():
    dates = pd.date_range("2025-07-07T15:00Z", periods=2, freq="1ns")
    analysis = PortfolioAnalysis.from_frames(
        pd.Series([0.01, 0.02], index=dates),
        transactions=pd.DataFrame(
            {"timestamp": dates, "asset": [7, 7], "quantity": [1, -1], "price": [100, -100]}
        ),
        valuations=pd.DataFrame(
            {
                "date": dates.insert(0, dates[0] - pd.Timedelta(1, "ns")),
                "equity": [1000, 1100, 1200],
            }
        ),
        calendar=None,
    )
    result = analysis.compute_transaction_analysis(incremental_bps=[10])
    assert result.timeline["date"].cast(pl.Int64).to_list() == dates.as_unit("ns").asi8.tolist()
    assert result.executions["asset"].to_list() == [7, 7]
    assert result.timeline["filled_notional"].to_list() == [100, 100]
    assert result.timeline["turnover"].to_list() == pytest.approx([0.1, 100 / 1100])
    assert result.metadata["calendar"] is None


def test_disabled_calendar_does_not_silently_match_execution_instants_to_date_labels():
    with pytest.raises(ValueError, match="match"):
        _account(calendar=None).compute_transaction_analysis()


def test_declared_timezone_resolves_naive_execution_instants_without_changing_stored_inputs():
    source = _account()
    fills = source.input_data.transactions.to_pandas()
    fills["timestamp"] = fills["timestamp"].dt.tz_localize(None)
    analysis = PortfolioAnalysis.from_frames(
        source.input_data.returns,
        return_col="return",
        transactions=fills,
        feed_spec={
            "timestamp_col": "date",
            "calendar": "CME_Equity",
            "timezone": "UTC",
            "timestamp_semantics": "session_label",
        },
        transaction_timestamp_col="timestamp",
    )
    result = analysis.compute_transaction_analysis()
    assert result.timeline["filled_notional"].to_list() == [200, 0, 0]
    assert analysis.input_data.transactions["timestamp"].dtype.time_zone is None


def test_midnight_bar_closes_use_completed_24h_sessions_not_next_session_opens():
    analysis = PortfolioAnalysis.from_frames(
        pd.Series([0.01, 0.02], index=pd.to_datetime(["2025-07-07T00:00Z", "2025-07-08T00:00Z"])),
        transactions=pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2025-07-06T12:00Z"]),
                "asset": ["A"],
                "quantity": [1.0],
                "price": [100.0],
            }
        ),
        feed_spec={
            "timestamp_col": "date",
            "calendar": "24/7",
            "time_contract_version": 2,
            "timestamp_semantics": "bar_close",
            "bar_label": "right",
        },
        transaction_timestamp_col="timestamp",
        periods_per_year=365,
    )
    result = analysis.compute_transaction_analysis()
    assert result.timeline["date"].to_list() == [
        pd.Timestamp("2025-07-06").date(),
        pd.Timestamp("2025-07-07").date(),
    ]
    assert result.timeline["filled_notional"].to_list() == [100.0, 0.0]


def test_midnight_actual_executions_belong_to_new_24h_session_even_with_bar_close_inputs():
    analysis = PortfolioAnalysis.from_frames(
        pd.Series([0.01, 0.02], index=pd.to_datetime(["2025-07-07T00:00Z", "2025-07-08T00:00Z"])),
        transactions=pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2025-07-07T00:00Z"]),
                "asset": ["A"],
                "quantity": [1.0],
                "price": [100.0],
            }
        ),
        feed_spec={
            "timestamp_col": "date",
            "calendar": "24/7",
            "time_contract_version": 2,
            "timestamp_semantics": "bar_close",
            "bar_label": "right",
        },
        transaction_timestamp_col="timestamp",
        periods_per_year=365,
    )
    assert analysis.compute_transaction_analysis().timeline["filled_notional"].to_list() == [
        0.0,
        100.0,
    ]


@pytest.mark.parametrize("instant", ["2025-07-06T20:00Z", "2025-07-07T21:30Z"])
def test_off_session_fills_are_not_mapped_to_a_different_or_civil_day(instant):
    account = _account()
    inputs = account.input_data
    fills = inputs.transactions.to_pandas()
    fills["timestamp"] = pd.to_datetime([instant] * 2)
    analysis = PortfolioAnalysis.from_frames(
        inputs.returns,
        return_col="return",
        transactions=fills,
        feed_spec={
            "timestamp_col": "date",
            "timestamp_semantics": "session_label",
            "calendar": "CME_Equity",
        },
        transaction_timestamp_col="timestamp",
    )
    with pytest.raises(ValueError, match="outside.*sessions"):
        analysis.compute_transaction_analysis()


def test_agb_uses_real_adjacent_complete_gross_snapshots_not_nav_or_net_exposure():
    account = _account()
    fills = account.input_data.transactions.to_pandas()
    fills["timestamp"] = pd.to_datetime(["2025-07-08T15:00Z", "2025-07-08T15:00Z"])
    analysis = PortfolioAnalysis.from_frames(
        account.input_data.returns,
        positions=account.input_data.positions,
        transactions=fills,
        valuations=account.input_data.valuations,
        return_col="return",
        complete_snapshots=True,
        feed_spec={
            "timestamp_col": "date",
            "timestamp_semantics": "session_label",
            "calendar": "CME_Equity",
        },
        transaction_timestamp_col="timestamp",
    )
    result = analysis.compute_transaction_analysis(
        turnover_denom="average_gross_book", turnover_method="one_way"
    )
    assert result.timeline["turnover"][1] == pytest.approx(200 / 300 / 2)
    assert result.scenarios.filter(pl.col("incremental_bps") == 10)["incremental_cost_return"][
        1
    ] == pytest.approx(0.2 / 1100)


def test_unreported_nav_keeps_active_cost_dollars_without_inventing_a_cost_return():
    source = _account()
    analysis = PortfolioAnalysis.from_frames(
        source.input_data.returns,
        return_col="return",
        transactions=source.input_data.transactions,
        feed_spec={
            "timestamp_col": "date",
            "timestamp_semantics": "session_label",
            "calendar": "CME_Equity",
        },
        transaction_timestamp_col="timestamp",
    )
    result = analysis.compute_transaction_analysis(incremental_bps=[10])
    assert result.timeline["turnover"].null_count() == 3
    assert result.scenarios["incremental_cost"].to_list() == [0.2, 0, 0]
    assert result.scenarios["net_return"].to_list() == [None, -0.02, 0.03]


def test_supplied_empty_execution_ledger_proves_zero_activity_on_the_entire_clock():
    source = _account()
    analysis = PortfolioAnalysis.from_frames(
        source.input_data.returns,
        return_col="return",
        transactions=source.input_data.transactions.clear(),
        feed_spec={
            "timestamp_col": "date",
            "timestamp_semantics": "session_label",
            "calendar": "CME_Equity",
        },
        transaction_timestamp_col="timestamp",
    )
    result = analysis.compute_transaction_analysis()
    assert result.metrics["num_fills"] == 0
    assert result.metrics["mean_turnover_active"] is None
    assert result.timeline["implementation_cost"].to_list() == [0, 0, 0]
    assert result.scenarios["net_return"].to_list() == [0.01, -0.02, 0.03] * 2
    timing = result.execution_timing()
    assert timing["num_fills"].sum() == 0
    assert timing["absolute_units"].sum() == 0
    assert timing["notional_fraction"].null_count() == timing.height


def test_real_portfolio_report_renders_prepared_transaction_values_and_keeps_owned_copy(tmp_path):
    pytest.importorskip("plotly")
    import json

    analysis = _account(unknown_cost=True)
    prepared = analysis.compute_transaction_analysis(incremental_bps=[0, 10])
    sheet = analysis.create_tear_sheet(transactions=prepared)
    assert sheet.availability["transactions"]["status"] == "available"
    assert list(sheet.figures["Actual Traded Notional"].data[0].y) == [200, 0, 0]
    assert list(sheet.figures["Transaction Turnover"].data[0].y) == [0.2, 0, 0]
    assert list(sheet.figures["Absolute Executed Units"].data[0].y) == [1, 0, 0]
    assert list(sheet.figures["Turnover Distribution"].data[0].x) == [0.2, 0, 0]
    assert list(sheet.figures["Turnover Distribution"].data[1].x) == [0.2]
    assert sum(sheet.figures["Execution Time Distribution"].data[0].y) == 1
    assert list(sheet.figures["Observed Execution Costs"].data[0].y) == [None, 0, 0]
    assert list(sheet.figures["Incremental Cost Return Scenarios"].data[1].y) == pytest.approx(
        [0.0098, -0.02, 0.03]
    )
    prepared.metadata["turnover_method"] = "incorrect mutation"
    assert sheet.transactions.metadata["turnover_method"] == "gross"
    exported = sheet.to_dict()["transactions"]
    # Exact execution instants and original return labels remain available, separate from sessions.
    assert "2025-07-06" in exported["executions"][0]["timestamp"]
    json.dumps(exported, default=str, allow_nan=False)
    path = tmp_path / "portfolio.html"
    sheet.save_html(path, include_plotlyjs=True)
    html = path.read_text(encoding="utf-8")
    assert "Actual Traded Notional" in html
    assert "subtract only extra bps" in html
    assert "Turnover denominator: beginning_nav" in html


def test_foreign_transaction_result_is_rejected_before_any_report_is_published(tmp_path):
    pytest.importorskip("plotly")
    analysis = _account()
    other = PortfolioAnalysis.from_frames(
        pd.Series([0.2], index=pd.date_range("2025-07-07", periods=1))
    )
    with pytest.raises(ValueError, match="exact.*observations"):
        other.create_tear_sheet(transactions=analysis.compute_transaction_analysis())
    assert not list(tmp_path.iterdir())


def test_published_pyfolio_agb_matches_adjacent_books_and_nav_cost_difference_is_explicit():
    from importlib.metadata import version

    txn = pytest.importorskip("pyfolio.txn")
    if version("pyfolio-reloaded") != "0.10.0":
        pytest.skip("the corrected reference profile is pinned to Pyfolio0.10.0")
    dates = pd.date_range("2025-07-07", periods=3, tz="UTC")
    returns = pd.Series([0.01, -0.02, 0.03], index=dates)
    legacy_positions = pd.DataFrame(
        {"A": [200.0, 400.0, 600.0], "cash": [900.0, 800.0, 700.0]}, index=dates
    )
    fills = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "amount": [1.0, -1.0],
            "price": [100.0, 100.0],
        },
        index=pd.to_datetime(["2025-07-08T15:00Z", "2025-07-08T16:00Z"]),
    )
    analysis = PortfolioAnalysis.from_frames(
        returns,
        positions=legacy_positions,
        complete_snapshots=True,
        valuations=pd.DataFrame(
            {
                "date": pd.DatetimeIndex(["2025-07-03T00:00Z", *dates]),
                "equity": [1000.0, 1100.0, 1200.0, 1300.0],
            }
        ),
        transactions=fills.rename(columns={"symbol": "asset", "amount": "quantity"})
        .rename_axis("timestamp")
        .reset_index(),
        transaction_timestamp_col="timestamp",
        feed_spec={
            "timestamp_col": "date",
            "timestamp_semantics": "session_label",
            "calendar": "NYSE",
        },
    )
    result = analysis.compute_transaction_analysis(
        turnover_denom="average_gross_book", incremental_bps=[10]
    )
    old = txn.get_turnover(legacy_positions, fills)
    assert old.iloc[1] == pytest.approx(200 / 300)
    assert result.timeline["turnover"][1] == pytest.approx(old.iloc[1])
    assert old.iloc[0] == 0 and result.timeline["turnover"][0] is None
    legacy_adjusted = txn.adjust_returns_for_slippage(returns, legacy_positions, fills, 10)
    assert legacy_adjusted.iloc[1] == pytest.approx(-0.02 - 0.2 / 1200)
    assert result.scenarios["net_return"][1] == pytest.approx(-0.02 - 0.2 / 1100)
    np.testing.assert_array_equal(analysis.returns, returns.to_numpy())
