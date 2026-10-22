"""Round-trip tables must keep distinct PnL, duration and NAV-return samples."""

import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.compat import pyfolio as pf
from ml4t.diagnostic.evaluation import PortfolioAnalysis
from ml4t.diagnostic.evaluation.portfolio_analysis import summarize_round_trips
from ml4t.diagnostic.evaluation.trade_analysis import TradeStatistics


def _partial_book():
    dates = pd.date_range("2025-07-07", periods=4, tz="UTC")
    fills = pd.DataFrame(
        {
            "timestamp": [dates[0], dates[1], dates[2], dates[2], dates[2], dates[3]],
            "asset": ["A", "A", "B", "C", "D", "D"],
            "quantity": [1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
            "price": [100.0, 120.0, 110.0, 120.0, 100.0, 100.0],
            "commission": [0.0] * 6,
        }
    )
    opening = pd.DataFrame(
        {
            "asset": ["B", "C"],
            "quantity": [-1.0, 1.0],
            "entry_price": [100.0, None],
            "entry_timestamp": pd.to_datetime([None, dates[0]]),
            "remaining_commission": [0.0, 0.0],
        }
    )
    analysis = PortfolioAnalysis.from_frames(
        pd.Series([0.01] * 4, index=dates), transactions=fills, calendar=None
    )
    result = analysis.compute_round_trips(
        initial_state="supplied", opening_lots=opening, opening_as_of=dates[0]
    )
    nav = pd.Series([1000.0, 1000.0], index=dates[[1, 3]])
    return result, nav


def _value(table, metric, scope="all"):
    return table.filter((pl.col("metric") == metric) & (pl.col("scope") == scope))["value"].item()


def _classic_frame():
    result, nav = _partial_book()
    return (
        result.portfolio_contributions(nav)
        .select(
            pl.col("gross_pnl").alias("pnl"),
            "duration",
            pl.col("asset").alias("symbol"),
            (pl.col("direction") == "long").alias("long"),
            pl.col("gross_nav_contribution").alias("returns"),
            pl.col("gross_lot_return").alias("rt_returns"),
        )
        .to_pandas()
    )


def test_actual_completed_statistics_need_no_invented_entry_prices_or_zero_unknown_durations():
    result, _ = _partial_book()
    frame = (
        result.completed_trades()
        .filter(pl.col("net_pnl").is_not_null())
        .select(
            pl.col("net_pnl").alias("pnl"),
            (pl.col("duration_seconds") / 86400).alias("duration_days"),
        )
    )
    before = frame.clone()
    stats = TradeStatistics.from_frame(frame)
    assert stats.n_trades == 3
    assert stats.total_pnl == 10.0
    assert stats.win_rate == 1 / 3
    assert stats.n_winners == stats.n_losers == 1
    assert stats.profit_factor == 2.0
    assert stats.gross_profit == 20.0
    assert stats.gross_loss == -10.0
    assert stats.min_pnl == -10.0
    assert stats.max_pnl == 20.0
    assert stats.n_known_durations == 2
    assert stats.avg_duration_days == 1.0
    assert frame.equals(before)
    unknown_time = TradeStatistics.from_frame(frame.select("pnl").to_pandas())
    assert unknown_time.total_pnl == 10.0
    assert unknown_time.n_known_durations == 0
    assert unknown_time.avg_duration_days is None


def test_unknown_cash_from_actual_unknown_opening_basis_is_not_zero_in_native_statistics():
    result, _ = _partial_book()
    cash = result.completed_trades().select(pl.col("gross_pnl").alias("pnl"))
    with pytest.raises(ValueError, match="pnl"):
        TradeStatistics.from_frame(cash)
    with pytest.raises(ValueError, match="gross_pnl"):
        result.statistics_tables(pnl="gross")


def test_real_partial_book_uses_independent_metric_masks_and_reports_all_coverage_counts():
    result, nav = _partial_book()
    summary = result.statistics_tables(
        pnl="gross", return_basis="nav", portfolio_value=nav, method="legacy_pyfolio"
    )
    assert _value(summary.tables["pnl"], "total_pnl") == 10.0
    assert _value(summary.tables["summary"], "n_trades") == 3.0
    assert _value(summary.tables["summary"], "win_rate") == 0.25
    assert _value(summary.tables["returns"], "mean") == 0.01
    assert summary.tables["duration"]["value"].dtype == pl.Duration("ns")
    duration = (
        summary.tables["duration"]
        .filter((pl.col("scope") == "all") & (pl.col("metric") == "mean"))["value"]
        .cast(pl.Int64)
        .item()
    )
    assert duration == pd.Timedelta("1D8h").value
    coverage = summary.coverage.filter(pl.col("scope") == "all").row(0, named=True)
    assert coverage["n_rows"] == 4
    assert coverage["n_known_pnl"] == 3
    assert coverage["n_known_duration"] == 3
    assert coverage["n_known_returns"] == 2
    assert coverage["n_finite_returns"] == 2
    assert summary.tables["symbols"]["asset"].unique().sort().to_list() == ["A", "B", "C", "D"]


def test_native_all_loss_profit_factor_definition_is_not_changed_by_explicit_legacy_tables():
    result, nav = _partial_book()
    frame = result.portfolio_contributions(nav).filter(pl.col("asset") == "B")
    native = summarize_round_trips(frame, pnl_col="gross_pnl", return_col="gross_nav_contribution")
    legacy = summarize_round_trips(
        frame, pnl_col="gross_pnl", return_col="gross_nav_contribution", method="legacy_pyfolio"
    )
    assert _value(native.tables["pnl"], "profit_factor") is None
    assert _value(legacy.tables["pnl"], "profit_factor") == 0.0
    assert _value(native.tables["pnl"], "total_pnl") == -10.0
    assert _value(native.tables["duration"], "mean") is None


def test_familiar_five_table_sample_does_not_drop_known_cash_for_missing_nav_or_time():
    source = _classic_frame()
    before = source.copy(deep=True)
    tables = pf.round_trips.gen_round_trip_stats(source)
    assert set(tables) == {"pnl", "summary", "duration", "returns", "symbols"}
    assert tables["pnl"].loc["Total profit", "All trades"] == 10.0
    assert tables["summary"].loc["Total number of round_trips", "All trades"] == 3.0
    assert tables["summary"].loc["Percent profitable", "All trades"] == 0.25
    assert tables["pnl"].loc["Profit factor", "Short trades"] == 0.0
    assert tables["duration"].loc["Avg duration", "All trades"] == pd.Timedelta("1D8h")
    assert tables["returns"].loc["Avg returns all round_trips", "All trades"] == 0.01
    assert pd.isna(tables["symbols"].loc["Avg returns all round_trips", "B"])
    assert tables["symbols"].columns.name == "symbol"
    assert tables["pnl"].attrs["diagnostic"]["method"] == "legacy_pyfolio"
    assert tables["pnl"].attrs["diagnostic"]["coverage"][0]["n_known_pnl"] == 3
    pd.testing.assert_frame_equal(source, before)


def test_all_five_real_published_pyfolio_tables_match_partial_book_values_labels_and_dtypes():
    reference = pytest.importorskip("pyfolio.round_trips")
    source = _classic_frame()
    expected = reference.gen_round_trip_stats(source)
    actual = pf.round_trips.gen_round_trip_stats(source)
    for name in ("pnl", "summary", "duration", "returns", "symbols"):
        pd.testing.assert_frame_equal(
            actual[name], expected[name], check_exact=False, rtol=2e-15, atol=1e-12
        )


def test_portfolio_return_tables_do_not_fall_back_to_available_lot_returns():
    source = _classic_frame().drop(columns="returns")
    assert "rt_returns" in source
    with pytest.raises(ValueError, match="portfolio.*returns"):
        pf.round_trips.gen_round_trip_stats(source)
    result, _ = _partial_book()
    with pytest.raises(ValueError, match="NAV.*portfolio_value"):
        result.statistics_tables(pnl="gross", return_basis="nav")


def test_typed_ns_durations_survive_extraction_and_mean_median_report_tables():
    dates = pd.to_datetime(
        [
            "2025-07-07T15:00:00.000000000Z",
            "2025-07-07T15:00:00.000000001Z",
            "2025-07-07T15:00:00.000000002Z",
            "2025-07-07T15:00:00.000000005Z",
        ]
    )
    fills = pd.DataFrame(
        {
            "amount": [1.0, -1.0, 1.0, -1.0],
            "symbol": ["A"] * 4,
            "price": [100.0, 110.0, 100.0, 120.0],
        },
        index=dates,
    )
    closed = pf.round_trips.extract_round_trips(
        fills, portfolio_value=pd.Series([1000.0], index=dates[:1].normalize())
    )
    tables = pf.round_trips.gen_round_trip_stats(closed)
    assert closed["duration"].tolist() == [pd.Timedelta(1, "ns"), pd.Timedelta(3, "ns")]
    assert tables["duration"].loc["Avg duration", "All trades"] == pd.Timedelta(2, "ns")
    assert tables["duration"].loc["Median duration", "All trades"] == pd.Timedelta(2, "ns")


def test_explicit_lot_table_basis_is_named_and_cannot_silently_ignore_supplied_nav():
    result, nav = _partial_book()
    frame = result.completed_trades().filter(pl.col("gross_pnl").is_not_null())
    lot = summarize_round_trips(frame, pnl_col="gross_pnl", return_col="gross_lot_return")
    assert lot.metadata["return_col"] == "gross_lot_return"
    with pytest.raises(ValueError, match="return_basis"):
        result.statistics_tables(pnl="gross", return_basis="lot", portfolio_value=nav)


def test_missingness_metadata_is_not_shared_mutable_state_between_returned_tables():
    tables = pf.round_trips.gen_round_trip_stats(_classic_frame())
    tables["pnl"].attrs["diagnostic"]["coverage"][0]["n_rows"] = 999
    assert tables["summary"].attrs["diagnostic"]["coverage"][0]["n_rows"] == 4


@pytest.mark.parametrize("nav_value", [0.0, -1000.0, None])
def test_real_legacy_nav_edge_arrays_keep_their_own_sign_and_missingness_in_all_tables(nav_value):
    reference = pytest.importorskip("pyfolio.round_trips")
    dates = pd.date_range("2025-07-07", periods=2, tz="UTC")
    fills = pd.DataFrame(
        {"symbol": ["A", "A"], "amount": [1, -1], "price": [100, 120]}, index=dates
    )
    nav = pd.Series([nav_value], index=dates[-1:], dtype=float)
    produced = pf.round_trips.extract_round_trips(fills, portfolio_value=nav)
    assert produced["pnl"].tolist() == [20.0]
    if nav_value == -1000.0:
        assert produced["returns"].tolist() == [-0.02]
    expected = reference.gen_round_trip_stats(produced)
    actual = pf.round_trips.gen_round_trip_stats(produced)
    for name in actual:
        pd.testing.assert_frame_equal(actual[name], expected[name])


def test_empty_extracted_book_has_no_invented_report_observations_or_zero_performance():
    dates = pd.date_range("2025-07-07", periods=1, tz="UTC")
    fills = pd.DataFrame({"symbol": ["A"], "amount": [1], "price": [100]}, index=dates)
    produced = pf.round_trips.extract_round_trips(
        fills, portfolio_value=pd.Series([1000.0], index=dates)
    )
    assert produced.empty
    tables = pf.round_trips.gen_round_trip_stats(produced)
    for table in tables.values():
        assert table.empty
        assert table.attrs["diagnostic"]["sample_state"] == "empty"
        assert table.attrs["diagnostic"]["coverage"] == []
