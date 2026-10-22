"""Retrospective estimates retain their real candidate sample without becoming actual account facts."""

from copy import deepcopy

import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.compat.pyfolio import utils as pf
from ml4t.diagnostic.evaluation import PortfolioAnalysis
from ml4t.diagnostic.evaluation.portfolio_analysis import estimate_intraday_positions


def _round_trip():
    dates = pd.to_datetime(["2025-07-07T00:00Z"])
    returns = pd.Series([0.004], index=dates)
    positions = pd.DataFrame({"A": [0.0], "cash": [1004.0]}, index=dates)
    tx = pd.DataFrame(
        {"symbol": ["A", "A"], "amount": [2.0, -2.0], "price": [100.0, 102.0]},
        index=pd.to_datetime(["2025-07-07T14:00Z", "2025-07-07T15:00Z"]),
    )
    return returns, positions, tx


def _analysis(returns, positions, tx, **kwargs):
    return PortfolioAnalysis.from_frames(
        returns,
        positions=positions,
        complete_snapshots=True,
        transactions=tx.rename(columns={"symbol": "asset", "amount": "quantity"}),
        calendar=None,
        **kwargs,
    )


def test_explicit_daily_estimate_keeps_selected_transaction_cash_separate_from_realized_cash_and_nav():
    returns, positions, tx = _round_trip()
    before = [deepcopy(frame) for frame in (returns, positions, tx)]
    analysis = _analysis(returns, positions, tx)
    result = analysis.compute_intraday_estimate(method="legacy_peak_notional")
    assert result.snapshot_kind == "estimated_intraday"
    assert result.timeline["gross_exposure"].to_list() == [200.0]
    assert result.timeline["cash"].to_list() == [800.0]
    assert result.timeline["equity"].to_list() == [1000.0]
    assert result.timeline["member_stop"].to_list() == [1]
    assert result.timeline["candidate_timestamp"].cast(pl.Int64).to_list() == [tx.index[0].value]
    assert result.timeline.schema["date"] == pl.Date
    assert "retrospective" in result.metadata["peak_semantics"]
    assert result.metadata["initial_capital"] == 1000.0
    assert analysis.compute_round_trips(initial_state="flat").metrics["realized_gross_pnl"] == 4.0
    assert analysis.input_data.valuations["cash"].to_list() == [1004.0]
    pd.testing.assert_series_equal(returns, before[0])
    pd.testing.assert_frame_equal(positions, before[1])
    pd.testing.assert_frame_equal(tx, before[2])


def test_all_equal_peak_candidates_remain_distinct_daily_rows_with_exact_clock_and_prefix_membership():
    returns, positions, _ = _round_trip()
    returns.iloc[0], positions.loc[:, "cash"] = 0.0, 1000.0
    times = pd.to_datetime(
        ["2025-07-07T14:00:00.000000007Z", "2025-07-07T15:00:00.000000009Z", "2025-07-07T16:00Z"],
        format="mixed",
    )
    tx = pd.DataFrame(
        {
            "symbol": ["A"] * 3,
            "amount": [1.0, -2.0, 1.0],
            "price": [100.0] * 3,
            "record_id": ["entry", "reverse", "exit"],
        },
        index=times,
    )
    result = _analysis(returns, positions, tx).compute_intraday_estimate(
        method="legacy_peak_notional"
    )
    assert result.timeline["gross_exposure"].to_list() == [100.0, 100.0]
    assert result.timeline["net_exposure"].to_list() == [100.0, -100.0]
    assert result.timeline["cash"].to_list() == [900.0, 1100.0]
    assert result.timeline["member_stop"].to_list() == [1, 2]
    assert result.timeline["observation_id"].to_list() == [0, 1]
    assert (
        result.timeline["candidate_timestamp"].cast(pl.Int64).to_list() == times[:2].asi8.tolist()
    )
    assert [row["record_id"] for row in result.metadata["execution_order"]] == [
        "entry",
        "reverse",
        "exit",
    ]
    familiar = pf.estimate_intraday(returns, positions, tx)
    assert familiar.index[0] == familiar.index[1]
    assert familiar["A"].to_list() == [100.0, -100.0]
    assert familiar.attrs["diagnostic"]["candidate_timestamps"] == [
        value.isoformat() for value in times[:2]
    ]


@pytest.mark.parametrize("same_instant", [False, True])
def test_simultaneous_fills_sum_cash_flow_instead_of_halving_exposure_through_pivot_mean(
    same_instant,
):
    returns, positions, _ = _round_trip()
    returns.iloc[0], positions.loc[:, "A"], positions.loc[:, "cash"] = 0.0, 200.0, 800.0
    tx = pd.DataFrame(
        {"symbol": ["A"] * 2, "amount": [1.0] * 2, "price": [100.0] * 2},
        index=pd.to_datetime(
            ["2025-07-07T14:00Z", "2025-07-07T14:00Z" if same_instant else "2025-07-07T14:01Z"]
        ),
    )
    result = _analysis(returns, positions, tx).compute_intraday_estimate(
        method="legacy_peak_notional"
    )
    assert result.timeline["gross_exposure"].to_list() == [200.0]
    assert result.timeline["member_stop"].to_list() == [2]
    assert pf.estimate_intraday(returns, positions, tx)["A"].to_list() == [200.0]


def test_legacy_nan_position_profile_is_explicit_and_does_not_relax_native_snapshot_validation():
    dates = pd.date_range("2025-07-07", periods=2, tz="UTC")
    returns = pd.Series([0.0, 0.0], index=dates)
    positions = pd.DataFrame({"A": [float("nan"), 100.0], "cash": [1000.0, 900.0]}, index=dates)
    tx = pd.DataFrame(
        {"symbol": ["A"], "amount": [1.0], "price": [100.0]},
        index=[dates[1] + pd.Timedelta(hours=14)],
    )
    with pytest.raises(ValueError, match="finite|missing"):
        _analysis(returns, positions, tx)
    result = pf.estimate_intraday(returns, positions, tx)
    assert result["A"].to_list() == [0.0, 100.0]
    assert result["cash"].to_list() == [1000.0, 900.0]
    assert result.attrs["diagnostic"]["legacy_position_missing_to_zero"] == 1
    assert pd.isna(positions["A"].iloc[0])


def test_unknown_first_return_does_not_invent_initial_cash_but_known_later_cash_is_retained():
    dates = pd.date_range("2025-07-07", periods=2, tz="UTC")
    returns = pd.Series([float("nan"), 0.0], index=dates)
    positions = pd.DataFrame({"A": [0.0, 0.0], "cash": [1000.0, 1000.0]}, index=dates)
    tx = pd.DataFrame(
        {"symbol": ["A", "A"], "amount": [1.0, -1.0], "price": [100.0, 100.0]},
        index=[dates[0] + pd.Timedelta(hours=14), dates[0] + pd.Timedelta(hours=15)],
    )
    with pytest.raises(ValueError, match="finite returns"):
        _analysis(returns, positions, tx)
    result = estimate_intraday_positions(
        positions,
        tx.rename(columns={"symbol": "asset", "amount": "quantity"}),
        first_return=None,
        complete_snapshots=True,
        calendar=None,
        method="legacy_peak_notional",
    )
    assert result.timeline["cash"].to_list() == [None, 1000.0]
    assert result.timeline["equity"].to_list() == [None, 1000.0]
    assert result.coverage["n_normalized_observations"] == 1
    assert result.timeline["candidate_timestamp"].null_count() == 1
    familiar = pf.estimate_intraday(returns, positions, tx)
    assert familiar["cash"].to_list() == [-100.0, 1000.0]
    assert familiar.attrs["diagnostic"]["capital_method"] == "legacy_zero_fill"
    assert pd.isna(returns.iloc[0])


def test_classifier_and_selector_keep_disabled_identity_missing_input_and_no_activity_contracts():
    import numpy as np

    returns, positions, tx = _round_trip()
    assert pf.detect_intraday(positions, tx)
    assert pf.check_intraday(False, returns, positions, tx) is positions
    assert pf.check_intraday(np.bool_(False), returns, positions, tx) is positions
    assert pf.check_intraday(None, returns, positions, tx) is positions
    assert pf.check_intraday("infer", returns, positions, None) is positions
    assert pf.check_intraday("infer", returns, None, tx) is None
    assert not pf.detect_intraday(positions, tx.iloc[:0])
    with pytest.warns(UserWarning, match="Detected intraday"):
        inferred = pf.check_intraday("infer", returns, positions, tx)
    pd.testing.assert_frame_equal(inferred, pf.check_intraday(True, returns, positions, tx))
    pd.testing.assert_frame_equal(
        inferred, pf.check_intraday(np.bool_(True), returns, positions, tx)
    )
    with pytest.raises(ValueError, match="needed"):
        pf.check_intraday(True, returns, positions, None)


def test_report_labels_the_daily_estimate_and_preserves_candidate_times_instead_of_promoting_eod_to_real_peaks(
    tmp_path,
):
    pytest.importorskip("plotly")
    from ml4t.diagnostic.visualization.portfolio import create_round_trip_dashboard

    returns, positions, tx = _round_trip()
    analysis = _analysis(returns, positions, tx)
    result = analysis.compute_intraday_estimate(method="legacy_peak_notional")
    report = create_round_trip_dashboard(
        analysis, analysis.compute_round_trips(initial_state="flat"), pnl="gross", intraday=result
    )
    assert "Estimated Intraday Holdings Exposure" in report.figures
    assert "Intraday Holdings Exposure" not in report.figures
    assert report.intraday.snapshot_kind == "estimated_intraday"
    exported = report.to_dict()["intraday"]
    assert exported["timeline"][0]["candidate_timestamp"] == tx.index[0].isoformat()
    assert (
        "not observed"
        in report.figures["Estimated Intraday Holdings Exposure"].layout.meta["description"]
    )
    report.save_html(tmp_path / "estimated.html", include_plotlyjs=True)


@pytest.mark.parametrize("eod_hour", [17, 23])
def test_actual_published_daily_estimate_and_ignored_eod_hour_match_on_shared_complete_book(
    eod_hour,
):
    legacy = pytest.importorskip("pyfolio.utils")
    returns, positions, tx = _round_trip()
    pd.testing.assert_frame_equal(
        pf.estimate_intraday(returns, positions, tx, EOD_hour=eod_hour),
        legacy.estimate_intraday(returns, positions, tx, EOD_hour=eod_hour),
    )
    assert pf.detect_intraday(positions, tx) == legacy.detect_intraday(positions, tx)


def test_empty_transaction_ledger_retains_carry_rows_without_fake_candidate_times():
    dates = pd.to_datetime(["2025-07-07T00:00Z"])
    positions = pd.DataFrame({"cash": [1000.0]}, index=dates)
    tx = pd.DataFrame(
        {
            "symbol": pd.Series(dtype=object),
            "amount": pd.Series(dtype=float),
            "price": pd.Series(dtype=float),
        },
        index=pd.DatetimeIndex([], tz="UTC"),
    )
    result = pf.estimate_intraday(pd.Series([0.0], index=dates), positions, tx)
    assert result.columns.to_list() == ["cash"] and result["cash"].to_list() == [1000.0]
    assert result.attrs["diagnostic"]["candidate_timestamps"] == [None]
    assert result.attrs["diagnostic"]["execution_order"] == []


def test_previous_eod_nontransacted_asset_is_carried_into_the_selected_candidate():
    dates = pd.date_range("2025-07-07", periods=2, tz="UTC")
    positions = pd.DataFrame(
        {"A": [0.0, 100.0], "B": [300.0, 300.0], "cash": [700.0, 600.0]}, index=dates
    )
    tx = pd.DataFrame(
        {"symbol": ["A"], "amount": [1.0], "price": [100.0]},
        index=[dates[1] + pd.Timedelta(hours=14)],
    )
    result = pf.estimate_intraday(pd.Series([0.0, 0.0], index=dates), positions, tx)
    assert result["A"].to_list() == [0.0, 100.0]
    assert result["B"].to_list() == [0.0, 300.0]
    assert result["cash"].to_list() == [1000.0, 600.0]


def test_familiar_projection_retains_distinct_integer_and_string_asset_ids():
    dates = pd.to_datetime(["2025-07-07T00:00Z"])
    positions = pd.DataFrame({1: [100.0], "1": [200.0], "cash": [700.0]}, index=dates)
    tx = pd.DataFrame(
        {"symbol": pd.Series([1, "1"], dtype=object), "amount": [1.0, 1.0], "price": [100.0, 200.0]}
    )
    tx.index = pd.to_datetime(["2025-07-07T14:00Z", "2025-07-07T15:00Z"])
    result = pf.estimate_intraday(pd.Series([0.0], index=dates), positions, tx)
    assert result[1].to_list() == [100.0] and result["1"].to_list() == [200.0]
    assert result["cash"].to_list() == [700.0]


def test_feed_column_declarations_and_explicit_calendar_none_are_respected():
    from ml4t.specs import FeedSpec

    returns, positions, tx = _round_trip()
    supplied_positions = positions.reset_index().rename(columns={"index": "when"})
    supplied_transactions = tx.reset_index().rename(
        columns={"index": "when", "symbol": "sec", "amount": "quantity"}
    )
    feed = FeedSpec(time_contract_version=2, timestamp_col="when", entity_col="sec", calendar=None)
    args = {
        "first_return": returns.iloc[0],
        "complete_snapshots": True,
        "method": "legacy_peak_notional",
        "feed_spec": feed,
    }
    inherited = estimate_intraday_positions(supplied_positions, supplied_transactions, **args)
    assert inherited.timeline["cash"].to_list() == [800.0]
    with pytest.raises(ValueError, match="calendar=None"):
        estimate_intraday_positions(
            supplied_positions, supplied_transactions, calendar="NYSE", **args
        )


def test_published_nan_capital_zero_fill_is_explicit_in_the_facade_not_the_native_default():
    legacy = pytest.importorskip("pyfolio.utils")
    dates = pd.date_range("2025-07-07", periods=2, tz="UTC")
    returns = pd.Series([float("nan"), 0.0], index=dates)
    positions = pd.DataFrame({"A": [0.0, 0.0], "cash": [1000.0, 1000.0]}, index=dates)
    tx = pd.DataFrame(
        {"symbol": ["A", "A"], "amount": [1.0, -1.0], "price": [100.0, 100.0]},
        index=[dates[0] + pd.Timedelta(hours=14), dates[0] + pd.Timedelta(hours=15)],
    )
    pd.testing.assert_frame_equal(
        pf.estimate_intraday(returns, positions, tx),
        legacy.estimate_intraday(returns, positions, tx),
    )
    native = estimate_intraday_positions(
        positions,
        tx.rename(columns={"symbol": "asset", "amount": "quantity"}),
        first_return=None,
        complete_snapshots=True,
        calendar=None,
        method="legacy_peak_notional",
    )
    assert native.timeline["cash"].to_list() == [None, 1000.0]
    assert native.metadata["capital_method"] == "defined_only"


def test_estimated_report_integer_id_display_is_lossless_without_changing_native_payload(tmp_path):
    import json

    pytest.importorskip("plotly")
    from ml4t.diagnostic.visualization.portfolio import create_round_trip_dashboard

    asset = 2**53 + 1
    day = pd.Timestamp("2025-07-07T00:00Z")
    analysis = PortfolioAnalysis.from_frames(
        pd.Series([0.004], index=[day]),
        calendar=None,
        positions=pd.DataFrame({"date": [day], "asset": [asset], "value": [0.0]}),
        valuations=pd.DataFrame({"date": [day], "cash": [1004.0]}),
        complete_snapshots=True,
        transactions=pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2025-07-07T14:00Z", "2025-07-07T15:00Z"]),
                "asset": [asset] * 2,
                "quantity": [2.0, -2.0],
                "price": [100.0, 102.0],
                "commission": [0.0, 0.0],
            }
        ),
    )
    estimate = analysis.compute_intraday_estimate(method="legacy_peak_notional")
    report = create_round_trip_dashboard(
        analysis, analysis.compute_round_trips(initial_state="flat"), intraday=estimate
    )
    figure = report.figures["Estimated Intraday Top Absolute Holdings"]
    assert list(figure.data[0].y) == [asset]
    output = tmp_path / "integer-estimate.html"
    report.save_html(output, include_plotlyjs=False)
    section = output.read_text(encoding="utf-8").split(
        "<h2>Estimated Intraday Top Absolute Holdings</h2>", 1
    )[1]
    call = section.split("Plotly.newPlot(", 1)[1].lstrip()
    decoder = json.JSONDecoder()
    _, end = decoder.raw_decode(call)
    traces, _ = decoder.raw_decode(call[end:].lstrip().removeprefix(",").lstrip())
    assert traces[0]["y"] == [str(asset)]
    assert list(figure.data[0].y) == [asset]
    assert report.intraday.positions["asset"].to_list() == [asset]
