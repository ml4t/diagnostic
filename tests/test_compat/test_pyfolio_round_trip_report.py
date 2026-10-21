"""Real daily account -> familiar complete round-trip sheet, without Pyfolio installed."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

pytest.importorskip("plotly")

from ml4t.diagnostic.compat import pyfolio as pf


def account():
    dates = pd.date_range("2025-07-07", periods=6, freq="B", tz="UTC")
    pnl = [2.0, -1.0, 3.0, -2.0, 4.0, -3.0]
    before = [1000.0, 1002.0, 1001.0, 1004.0, 1002.0, 1006.0]
    after = [1002.0, 1001.0, 1004.0, 1002.0, 1006.0, 1003.0]
    returns = pd.Series([p / nav for p, nav in zip(pnl, before, strict=True)], index=dates)
    positions = pd.DataFrame({"A": 0.0, "cash": after}, index=dates)
    rows, times = [], []
    for date, profit in zip(dates, pnl, strict=True):
        rows.extend(
            [
                {"symbol": "A", "amount": 1.0, "price": 100.0},
                {"symbol": "A", "amount": -1.0, "price": 100.0 + profit},
            ]
        )
        times.extend([date + pd.Timedelta(hours=14), date + pd.Timedelta(hours=15)])
    return returns, positions, pd.DataFrame(rows, index=pd.DatetimeIndex(times))


def test_complete_familiar_sheet_uses_actual_inventory_and_supplied_nav():
    returns, positions, transactions = account()
    original = positions.copy(deep=True)
    report = pf.tears.create_round_trip_tear_sheet(
        returns,
        positions,
        transactions,
        sector_mappings={"A": "Supplied"},
        estimate_intraday=False,
        return_fig=True,
    )
    assert report.metrics is None
    assert report.round_trips.completed_trades().height == 6
    assert report.round_trips.executions.height == 12
    assert report.round_trips.open_lots.is_empty()
    stats = report.round_trip_statistics
    assert stats.observations["gross_pnl"].to_list() == [2.0, -1.0, 3.0, -2.0, 4.0, -3.0]
    assert stats.observations["gross_nav_contribution"][0] == pytest.approx(0.002)
    posterior = report.figures["Round-trip Win Probability"].layout.meta["posterior"]
    assert posterior["posterior_alpha"] == posterior["posterior_beta"] == 3
    assert set(report.figures) == {
        "Round-trip Lifetimes",
        "Round-trip Win Probability",
        "Round-trip Holding Duration",
        "Round-trip Cash PnL",
        "Round-trip Returns",
        "Round-trip Asset Attribution",
        "Round-trip Sector Attribution",
    }
    pd.testing.assert_frame_equal(positions, original)


def test_inferred_peak_is_not_used_as_actual_terminal_inventory():
    returns, positions, transactions = account()
    with pytest.warns(UserWarning, match="intraday"):
        report = pf.tears.create_round_trip_tear_sheet(
            returns, positions, transactions, return_fig=True
        )
    assert report.round_trips.executions.height == 12
    assert report.round_trips.open_lots.is_empty()
    assert report.round_trips.synthetic_segments is None
    assert (
        report.round_trip_statistics.metadata["intraday_estimation"]["snapshot_kind"]
        == "estimated_intraday"
    )


def test_forced_estimate_does_not_change_the_hypothetical_closing_price():
    date = pd.DatetimeIndex(["2025-07-07"], tz="UTC")
    returns = pd.Series([0.01], index=date)
    positions = pd.DataFrame({"A": [110.0], "cash": [900.0]}, index=date)
    transactions = pd.DataFrame(
        {"symbol": ["A"], "amount": [1.0], "price": [100.0]}, index=date + pd.Timedelta(hours=14)
    )
    with pytest.warns(UserWarning, match="Fewer"):
        report = pf.tears.create_round_trip_tear_sheet(
            returns, positions, transactions, estimate_intraday=True, return_fig=True
        )
    assert report.round_trips.executions.height == 1
    assert report.round_trips.segments.is_empty()
    assert report.round_trips.open_lots.height == 1
    assert report.round_trips.synthetic_segments["gross_pnl"].to_list() == [10.0]
    assert report.round_trip_statistics.observations["is_synthetic"].to_list() == [True]
    assert list(report.figures["Round-trip Returns"].data[0].x) == pytest.approx([1.01])
    assert "estimated_intraday" == report.round_trip_statistics.metadata["denominator"]["source"]


def test_unknown_nav_does_not_drop_known_cash_except_explicit_sector_profile():
    returns, positions, transactions = account()
    returns.iloc[0] = float("nan")
    report = pf.tears.create_round_trip_tear_sheet(
        returns,
        positions,
        transactions,
        sector_mappings={"A": "Supplied"},
        estimate_intraday=False,
        return_fig=True,
    )
    observations = report.round_trip_statistics.observations
    assert observations.height == 6
    assert observations["gross_nav_contribution"][0] is None
    assert observations["gross_pnl"].sum() == 3
    attribution = report.round_trip_attribution
    assert attribution.filter(pl.col("scope") == "asset")["pnl"].to_list() == [3.0]
    assert attribution.filter(pl.col("scope") == "sector")["pnl"].to_list() == [1.0]
    assert report.round_trip_statistics.metadata["sector_projection"]["n_output_rows"] == 5


def test_execution_only_report_exports_without_inventing_performance(tmp_path):
    report = pf.tears.create_round_trip_tear_sheet(
        *account(), estimate_intraday=False, return_fig=True
    )
    output = tmp_path / "round-trips.html"
    report.save_html(output, include_plotlyjs=True)
    rendered = output.read_text(encoding="utf-8")
    assert "Round-trip statistics" in rendered
    assert "Performance Summary" not in rendered
    assert report.to_dict()["metrics"] is None
    assert report.to_dict()["round_trips"]["metadata"]["compatibility_profile"]["asset_labels"] == [
        "A"
    ]
    assert list(report.figures["Round-trip Holding Duration"].data[0].x) == [0.0] * 6


def test_default_display_returns_none_and_keeps_the_complete_report(monkeypatch):
    from ml4t.diagnostic.visualization.portfolio import PortfolioTearSheet

    displayed = []
    monkeypatch.setattr(PortfolioTearSheet, "show", lambda report: displayed.append(report))
    result = pf.tears.create_round_trip_tear_sheet(*account(), estimate_intraday=False)
    assert result is None
    assert len(displayed) == 1
    assert displayed[0].round_trip_statistics.observations.height == 6


def test_familiar_lifetime_sample_preserves_ns_and_does_not_reset_global_rng():
    import plotly.graph_objects as go

    _, _, transactions = account()
    trades = pf.round_trips.extract_round_trips(transactions)
    trades = pd.concat(
        [trades.assign(symbol=f"A{index}") for index in range(20)], ignore_index=True
    )
    trades.loc[0, "open_dt"] += pd.Timedelta(7, unit="ns")
    original = trades.copy(deep=True)
    state = np.random.get_state()
    figure = pf.plotting.plot_round_trip_lifetimes(trades, disp_amount=20, lsize=9, ax=go.Figure())
    after = np.random.get_state()
    assert state[0] == after[0] and np.array_equal(state[1], after[1]) and state[2:] == after[2:]
    assert len(figure.layout.yaxis.ticktext) == 20
    assert any(str(trace.x[0]).endswith("000000007+00:00") for trace in figure.data)
    assert all(trace.line.width == 9 for trace in figure.data)
    pd.testing.assert_frame_equal(trades, original)
    sample = pf.plotting.plot_round_trip_lifetimes(trades, disp_amount=3)
    expected = np.random.RandomState(1).choice(20, size=3, replace=False)
    assert list(sample.layout.yaxis.ticktext) == [f"A{index}" for index in expected]


@pytest.mark.parametrize("profits", [[], [1.0], [-1.0], [0.0]])
def test_zero_prior_probability_marks_undefined_samples_instead_of_plotting_nan(profits):
    trades = pd.DataFrame({"pnl": pd.Series(profits, dtype=float)})
    figure = pf.plotting.plot_prob_profit_trade(trades)
    assert not figure.data
    assert figure.layout.meta["posterior"]["sample_state"] == "undefined_legacy_posterior"
    assert figure.layout.meta["posterior"]["mean"] is None
    assert "Undefined" in figure.layout.annotations[0].text


def test_all_cash_zero_execution_sheet_is_empty_not_a_fabricated_trade():
    dates = pd.DatetimeIndex(["2025-07-07"], tz="UTC")
    transactions = pd.DataFrame(
        {
            "symbol": pd.Series(dtype=str),
            "amount": pd.Series(dtype=float),
            "price": pd.Series(dtype=float),
        },
        index=pd.DatetimeIndex([], tz="UTC"),
    )
    with pytest.warns(UserWarning, match="Fewer"):
        report = pf.tears.create_round_trip_tear_sheet(
            pd.Series([float("nan")], index=dates),
            pd.DataFrame({"cash": [1000.0]}, index=dates),
            transactions,
            estimate_intraday=False,
            return_fig=True,
            sector_mappings={},
        )
    assert report.round_trip_statistics.observations.is_empty()
    assert report.round_trips.executions.is_empty()
    assert report.metrics is None
    assert (
        report.figures["Round-trip Win Probability"].layout.meta["posterior"]["sample_state"]
        == "undefined_legacy_posterior"
    )


def test_printed_cash_attribution_uses_per_metric_missingness_and_retains_label_types(capsys):
    trades = pd.DataFrame(
        {"symbol": pd.Series([1, "1", 1], dtype=object), "pnl": [10.0, -5.0, float("nan")]}
    )
    pf.plotting.show_profit_attribution(trades)
    lines = capsys.readouterr().out.splitlines()
    assert any("200.00%" in line for line in lines)
    assert any("-100.00%" in line for line in lines)
    assert len(lines) == 3


def test_tied_peak_candidates_do_not_duplicate_actual_closing_observations():
    dates = pd.DatetimeIndex(["2025-07-07"], tz="UTC")
    transactions = pd.DataFrame(
        {"symbol": ["A"] * 3, "amount": [1.0, -2.0, 1.0], "price": [100.0] * 3},
        index=pd.to_datetime(["2025-07-07T14:00Z", "2025-07-07T15:00Z", "2025-07-07T16:00Z"]),
    )
    with pytest.warns(UserWarning, match="Fewer"):
        report = pf.tears.create_round_trip_tear_sheet(
            pd.Series([0.0], index=dates),
            pd.DataFrame({"A": [0.0], "cash": [1000.0]}, index=dates),
            transactions,
            estimate_intraday=True,
            return_fig=True,
        )
    statistics = report.round_trip_statistics
    assert statistics.observations.height == 2
    assert statistics.observations["gross_pnl"].to_list() == [0.0, 0.0]
    assert len(statistics.metadata["intraday_estimation"]["candidate_timestamps"]) == 2
    assert statistics.metadata["denominator"]["equal_daily_candidate_duplicates_collapsed"] == 1


def test_real_published_probability_and_lifetime_selection_are_compared():
    legacy = pytest.importorskip("pyfolio.plotting")
    _, _, transactions = account()
    trades = pf.round_trips.extract_round_trips(transactions)
    reference = legacy.plot_prob_profit_trade(trades.copy(deep=True))
    figure = pf.plotting.plot_prob_profit_trade(trades)
    np.testing.assert_array_equal(figure.data[0].x, reference.lines[0].get_xdata())
    np.testing.assert_allclose(figure.data[0].y, reference.lines[0].get_ydata(), rtol=1e-14)
    np.testing.assert_allclose(figure.layout.xaxis.range, reference.get_xlim(), rtol=1e-14)
    import matplotlib.pyplot as plt

    plt.close(reference.figure)
    native = pf.plotting.plot_round_trip_lifetimes(trades)
    reference = legacy.plot_round_trip_lifetimes(trades.copy(deep=True))
    assert list(native.layout.yaxis.ticktext) == [
        text.get_text() for text in reference.get_yticklabels()
    ]
    plt.close(reference.figure)


def test_real_published_full_sheet_runs_on_the_same_complete_account():
    legacy = pytest.importorskip("pyfolio.tears")
    import matplotlib.pyplot as plt

    returns, positions, transactions = account()
    reference = legacy.create_round_trip_tear_sheet(
        returns, positions, transactions, estimate_intraday=False, return_fig=True
    )
    assert len(reference.axes) == 5
    report = pf.tears.create_round_trip_tear_sheet(
        returns, positions, transactions, estimate_intraday=False, return_fig=True
    )
    assert len(report.figures) == 6
    expected = pd.Series([0.002, -1 / 1002, 3 / 1001, -2 / 1004, 4 / 1002, -3 / 1006])
    np.testing.assert_allclose(
        report.round_trip_statistics.observations["gross_nav_contribution"].to_numpy(), expected
    )
    plt.close(reference)


def test_real_published_tied_candidates_and_native_keep_two_actual_closes():
    legacy_utils = pytest.importorskip("pyfolio.utils")
    legacy_round_trips = pytest.importorskip("pyfolio.round_trips")
    dates = pd.DatetimeIndex(["2025-07-07"], tz="UTC")
    returns = pd.Series([0.0], index=dates)
    positions = pd.DataFrame({"A": [0.0], "cash": [1000.0]}, index=dates)
    transactions = pd.DataFrame(
        {"symbol": ["A"] * 3, "amount": [1.0, -2.0, 1.0], "price": [100.0] * 3},
        index=pd.to_datetime(["2025-07-07T14:00Z", "2025-07-07T15:00Z", "2025-07-07T16:00Z"]),
    )
    estimate = legacy_utils.estimate_intraday(returns, positions, transactions)
    assert len(estimate) == 2
    reference = legacy_round_trips.extract_round_trips(
        transactions, portfolio_value=estimate.sum(axis=1) / (1 + returns)
    )
    assert len(reference) == 2
    with pytest.warns(UserWarning, match="Fewer"):
        report = pf.tears.create_round_trip_tear_sheet(
            returns, positions, transactions, estimate_intraday=True, return_fig=True
        )
    # Sell2 closes one long and opens one short; final buy1 closes that short.
    assert report.round_trip_statistics.observations.height == 2


def test_actual_reference_one_sided_density_crashes_on_real_extracted_trades():
    legacy = pytest.importorskip("pyfolio.plotting")
    _, _, transactions = account()
    selling = transactions["amount"] < 0
    transactions.loc[selling, "price"] = 100 + (transactions.loc[selling, "price"] - 100).abs()
    trades = pf.round_trips.extract_round_trips(transactions)
    assert len(trades) == 6 and (trades["pnl"] > 0).all()
    with pytest.raises(ValueError, match="NaN|Inf"):
        legacy.plot_prob_profit_trade(trades.copy(deep=True))
    figure = pf.plotting.plot_prob_profit_trade(trades)
    assert figure.layout.meta["posterior"]["posterior_alpha"] == 6
    assert figure.layout.meta["posterior"]["posterior_beta"] == 0
    assert not figure.data
    import matplotlib.pyplot as plt

    plt.close("all")
