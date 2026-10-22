"""Closing-fill samples must not count entry-lot segments as independent trades."""

import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.evaluation import PortfolioAnalysis
from ml4t.diagnostic.evaluation.trade_analysis import TradeAnalysis, TradeFilters, TradeMetrics


def _account(quantities=(10.0, 5.0, -12.0), prices=(100.0, 110.0, 120.0), fees=(10.0, 10.0, 12.0)):
    dates = pd.date_range("2025-07-07", periods=len(quantities), tz="UTC")
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
            }
        ),
    )


def test_one_closing_fill_is_one_observation_and_its_return_is_not_mean_segment_returns():
    result = _account().compute_round_trips(initial_state="flat")
    completed = result.completed_trades()
    assert result.segments.height == 2
    assert completed.height == 1
    assert completed["quantity"].to_list() == [12.0]
    assert completed["entry_notional"].to_list() == [1220.0]
    assert completed["gross_lot_return"].to_list() == [pytest.approx(220 / 1220)]
    assert completed["net_lot_return"].to_list() == [pytest.approx(194 / 1220)]
    assert completed["n_segments"].to_list() == [2]
    assert completed["exit_record_id"].to_list() == [2]


def test_existing_statistics_use_one_closing_fill_and_explicit_gross_or_net_cash_view():
    result = _account().compute_round_trips(initial_state="flat")
    native = result.to_trade_analysis(pnl="net")
    assert isinstance(native, TradeAnalysis)
    assert native.compute_statistics().n_trades == 1
    assert native.compute_statistics().total_pnl == 194.0
    assert native.trades[0].return_pct == pytest.approx(194 / 1220)
    assert result.to_trade_analysis(pnl="gross").compute_statistics().total_pnl == 220.0
    assert native.source_metadata["n_excluded_unknown_pnl"] == 0
    assert native.trades[0].metadata["asset"] == 7


def test_missing_fees_cannot_silently_select_zero_fee_net_pnl():
    result = _account(fees=(None, None, None)).compute_round_trips(initial_state="flat")
    with pytest.raises(ValueError, match="unknown|missing"):
        result.to_trade_analysis()
    assert result.to_trade_analysis(pnl="gross").compute_statistics().total_pnl == 220.0
    assert result.win_probability(missing="exclude")["n_excluded_unknown_pnl"] == 1


def test_one_win_beta11_has_independently_calculable_beta21_mean_and_quantiles():
    posterior = (
        _account().compute_round_trips(initial_state="flat").win_probability(confidence=0.95)
    )
    assert posterior["n_observations"] == 1
    assert posterior["prior_alpha"] == posterior["prior_beta"] == 1.0
    assert posterior["posterior_alpha"] == 2.0
    assert posterior["posterior_beta"] == 1.0
    assert posterior["mean"] == pytest.approx(2 / 3)
    assert posterior["lower"] == pytest.approx(0.025**0.5)
    assert posterior["upper"] == pytest.approx(0.975**0.5)


def _unknown_entry_time():
    dates = pd.DatetimeIndex(["2025-07-07T00:00Z"])
    analysis = PortfolioAnalysis.from_frames(
        pd.Series([0.01], index=dates),
        calendar=None,
        transactions=pd.DataFrame(
            {
                "timestamp": dates,
                "asset": [7],
                "quantity": [-1.0],
                "price": [110.0],
                "commission": [0.0],
            }
        ),
    )
    return analysis.compute_round_trips(
        initial_state="supplied",
        opening_as_of=dates[0],
        opening_lots=pd.DataFrame(
            {"asset": [7], "quantity": [1.0], "entry_price": [100.0], "remaining_commission": [0.0]}
        ),
    )


def test_known_pnl_from_actual_opening_producer_with_unknown_time_remains_in_statistics():
    result = _unknown_entry_time()
    row = result.segments.row(0, named=True)
    metric = TradeMetrics(
        timestamp=row["exit_timestamp"],
        symbol="7",
        entry_price=row["entry_price"],
        exit_price=row["exit_price"],
        pnl=row["net_pnl"],
        duration=None,
        direction="long",
        quantity=row["quantity"],
    )
    stats = TradeAnalysis([metric]).compute_statistics()
    assert stats.total_pnl == 10.0
    assert stats.avg_duration_days is None
    assert stats.n_known_durations == 0
    assert metric.pnl_per_day is None
    automatic = result.to_trade_analysis()
    assert automatic.compute_statistics().total_pnl == 10.0
    assert automatic.compute_statistics().avg_duration_days is None
    assert "unknown" in automatic.compute_statistics().summary()
    assert automatic.trades[0].duration is None
    restored = TradeMetrics.model_validate_json(automatic.trades[0].model_dump_json())
    assert restored.duration is None
    assert restored.return_pct == 0.1


def test_minimum_duration_filter_excludes_unknown_duration_instead_of_inventing_zero_days():
    metrics = _unknown_entry_time().to_trade_analysis().trades
    with pytest.raises(ValueError, match="No trades"):
        TradeAnalysis(metrics, filters=TradeFilters(min_duration=pd.Timedelta(0)))


def test_default_native_price_return_does_not_change_when_fifo_analytical_basis_is_added():
    trade = _account().compute_round_trips(initial_state="flat").to_trade_analysis().trades[0]
    assert trade.return_basis == "entry_notional"
    native = trade.model_copy(update={"return_basis": "price"})
    assert native.return_pct == pytest.approx((120 - (1220 / 12)) / (1220 / 12))
    assert native.return_pct != pytest.approx(trade.return_pct)


def test_break_even_is_nonwin_in_beta_model_and_no_closes_retains_declared_uniform_prior():
    result = _account((1.0, -1.0), (100.0, 100.0), (0.0, 0.0)).compute_round_trips(
        initial_state="flat"
    )
    posterior = result.win_probability()
    assert posterior["n_even"] == posterior["n_nonwinners"] == 1
    assert posterior["mean"] == pytest.approx(1 / 3)
    prior = (
        _account((1.0,), (100.0,), (0.0,))
        .compute_round_trips(initial_state="flat")
        .win_probability()
    )
    assert prior["sample_state"] == "prior_only"
    assert prior["mean"] == 0.5
    assert prior["lower"] == pytest.approx(0.025)
    assert prior["upper"] == pytest.approx(0.975)


def test_synthetic_observations_are_excluded_from_actual_sample_unless_explicitly_selected():
    result = _account().compute_round_trips(
        initial_state="flat",
        as_of="2025-07-09T22:00Z",
        marks=pd.DataFrame(
            {"timestamp": pd.to_datetime(["2025-07-09T21:00Z"]), "asset": [7], "price": [130]}
        ),
        synthetic_close=True,
    )
    assert result.win_probability()["n_observations"] == 1
    combined = result.completed_trades(include_synthetic=True)
    assert combined["is_synthetic"].to_list() == [False, True]
    assert combined["net_pnl"].to_list() == [194.0, 54.0]
    assert result.win_probability(include_synthetic=True)["n_observations"] == 2
    assert result.to_trade_analysis(include_synthetic=True).compute_statistics().total_pnl == 248.0
    assert result.metrics["realized_net_pnl"] == 194.0


@pytest.mark.parametrize("confidence", [True, 0.0, 1.0, -1.0, float("nan"), float("inf")])
def test_invalid_beta_interval_confidence_is_rejected(confidence):
    with pytest.raises(ValueError, match="confidence"):
        _account().compute_round_trips(initial_state="flat").win_probability(confidence=confidence)


def test_aggregate_preserves_ns_instants_and_exact_duration_in_existing_tables_and_json():
    dates = pd.to_datetime(["2025-07-07T00:00:00.000000001Z", "2025-07-07T00:00:00.000000003Z"])
    analysis = PortfolioAnalysis.from_frames(
        pd.Series([0.01, 0.02], index=dates),
        calendar=None,
        transactions=pd.DataFrame(
            {
                "timestamp": dates,
                "asset": [7, 7],
                "quantity": [1.0, -1.0],
                "price": [100.0, 110.0],
                "commission": [0.0, 0.0],
            }
        ),
    )
    result = analysis.compute_round_trips(initial_state="flat")
    metric = result.to_trade_analysis().trades[0]
    assert metric.duration_seconds == 2e-9
    assert metric.entry_timestamp.value == dates[0].value
    frame = TradeMetrics.to_dataframe([metric])
    assert frame["timestamp"].cast(pl.Int64).to_list() == [dates[1].value]
    restored = TradeMetrics.model_validate_json(metric.model_dump_json())
    assert restored.duration_seconds == 2e-9


def test_known_duration_summary_counts_only_available_times_without_losing_other_known_pnl():
    unknown = _unknown_entry_time().to_trade_analysis().trades[0]
    known = _account().compute_round_trips(initial_state="flat").to_trade_analysis().trades[0]
    stats = TradeAnalysis([unknown, known]).compute_statistics()
    assert stats.n_trades == 2
    assert stats.total_pnl == 204.0
    assert stats.n_known_durations == 1
    assert stats.avg_duration_days == 2.0


def test_aggregation_sums_absolute_entry_notionals_when_positive_and_negative_price_lots_offset():
    result = _account(
        (1.0, 1.0, -2.0), (-100.0, 100.0, 110.0), (0.0, 0.0, 0.0)
    ).compute_round_trips(initial_state="flat")
    row = result.completed_trades().row(0, named=True)
    assert row["entry_price"] == 0.0
    assert row["entry_notional"] == 200.0
    assert row["gross_lot_return"] == 1.1
    assert result.to_trade_analysis().trades[0].return_pct == 1.1


def test_partial_missing_basis_is_retained_in_table_and_only_explicitly_excluded_from_sample():
    result = _unknown_entry_time()
    # Re-run the same producing account with unknown cost basis, not a fabricated result row.
    dates = pd.DatetimeIndex(["2025-07-07T00:00Z"])
    source = PortfolioAnalysis.from_frames(
        pd.Series([0.01], index=dates), calendar=None, transactions=result.executions
    )
    unknown = source.compute_round_trips(
        initial_state="supplied",
        opening_as_of=dates[0],
        opening_lots=pd.DataFrame({"asset": [7], "quantity": [1.0]}),
    )
    assert unknown.completed_trades()["gross_pnl"].to_list() == [None]
    with pytest.raises(ValueError, match="unknown"):
        unknown.win_probability(pnl="gross")
    prior = unknown.win_probability(pnl="gross", missing="exclude")
    assert prior["n_observations"] == 0
    assert prior["n_excluded_unknown_pnl"] == 1


def test_published_pyfolio_closing_fill_matches_native_count_cash_basis_and_lot_return():
    reference = pytest.importorskip("pyfolio.round_trips")
    result = _account(fees=(0.0, 0.0, 0.0)).compute_round_trips(initial_state="flat")
    fills = (
        result.executions.to_pandas()
        .set_index("timestamp")
        .rename(columns={"asset": "symbol", "quantity": "amount"})
    )
    fills["symbol"] = fills["symbol"].map({7: "A"})
    legacy = reference.extract_round_trips(fills)
    native = result.completed_trades()
    assert len(legacy) == native.height == 1
    assert legacy["pnl"].to_list() == native["gross_pnl"].to_list() == [220.0]
    assert legacy["rt_returns"].to_list() == native["gross_lot_return"].to_list()
    assert native["entry_notional"].to_list() == [1220.0]


def test_closing_fill_sums_exact_cash_before_large_offsetting_segment_pnl_is_rounded():
    result = _account((1, 1, -2), (0, 2 * 10**16, 10**16 + 1), (0.0, 0.0, 0.0)).compute_round_trips(
        initial_state="flat"
    )
    assert result.metrics["realized_gross_pnl"] == 2.0
    assert result.completed_trades()["gross_pnl"].to_list() == [2.0]
    assert result.to_trade_analysis().compute_statistics().total_pnl == 2.0


def test_single_or_constant_pnl_has_undefined_moments_instead_of_scipy_nan_or_warnings():
    stats = (
        _account()
        .compute_round_trips(initial_state="flat")
        .to_trade_analysis()
        .compute_statistics()
    )
    assert stats.pnl_skewness is None
    assert stats.pnl_kurtosis is None
