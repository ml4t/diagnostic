"""Complete account manifests must preserve supplied returns and nested report options."""

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.compat import pyfolio as pf


def _account():
    dates = pd.bdate_range("2025-02-03", periods=4, tz="UTC")
    prices = pd.Series([100.0, 110, 99, 103.95], index=dates)
    returns = prices.pct_change(fill_method=None).iloc[1:]
    nav = 1000 * prices.iloc[1:] / prices.iloc[0]
    positions = pd.DataFrame({"PRIVATE_A": 0.6 * nav, "PRIVATE_B": -0.2 * nav, "cash": 0.6 * nav})
    fills = pd.DataFrame(
        {"symbol": ["PRIVATE_A"] * 2, "amount": [1.0, -1], "price": [110.0, 99]},
        index=returns.index[:2] + pd.Timedelta(hours=15),
    )
    return returns, positions, fills


def _figure(report, title):
    matches = [figure for name, figure in report.figures.items() if name.split("/")[-1] == title]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize("name", ["create_simple_tear_sheet", "create_full_tear_sheet"])
def test_complete_account_workflow_preserves_price_produced_returns_and_actual_fill_samples(
    name, tmp_path
):
    returns, positions, fills = _account()
    original = [value.copy(deep=True) for value in (returns, positions, fills)]
    report = getattr(pf, name)(
        returns, positions=positions, transactions=fills, estimate_intraday=False, return_fig=True
    )
    assert _figure(report, "Cumulative wealth").data[0].y[-1] == pytest.approx(1.0395)
    np.testing.assert_allclose(
        _figure(report, "Turnover").data[0].y, [110 / 440, 99 / 836, 0], rtol=1e-12
    )
    assert report.availability["composition"]["status"] == "available"
    report.save_html(tmp_path / f"{name}.html", include_plotlyjs=True)
    for before, after in zip(original, (returns, positions, fills), strict=True):
        if isinstance(before, pd.Series):
            pd.testing.assert_series_equal(before, after)
        else:
            pd.testing.assert_frame_equal(before, after)


@pytest.mark.parametrize("name", ["create_simple_tear_sheet", "create_full_tear_sheet"])
def test_account_report_passes_closing_value_turnover_to_every_selected_component(name):
    returns, positions, fills = _account()
    report = getattr(pf, name)(
        returns,
        positions=positions,
        transactions=fills,
        turnover_denom="portfolio_value",
        estimate_intraday=False,
        return_fig=True,
    )
    np.testing.assert_allclose(_figure(report, "Turnover").data[0].y, [0.1, 0.1, 0])
    assert _figure(report, "Turnover").layout.meta["denominator"] == "portfolio_value"


def test_full_returns_only_route_does_not_require_a_producer_or_invent_an_empty_transaction_ledger():
    returns, _, _ = _account()
    report = pf.create_full_tear_sheet(returns, return_fig=True)
    assert report.availability["positions"]["status"] == "not_requested"
    assert report.availability["transactions"]["status"] == "not_requested"
    assert not any("Turnover" in name for name in report.figures)


def test_simple_slippage_compounds_known_additional_cost_once_instead_of_reconstructing_returns():
    returns, positions, fills = _account()
    report = pf.create_simple_tear_sheet(
        returns,
        positions=positions,
        transactions=fills,
        slippage=10,
        estimate_intraday=False,
        return_fig=True,
    )
    assert _figure(report, "Cumulative wealth").data[0].y[-1] == pytest.approx(1.0392900105)


def test_full_annualization_reaches_returns_rolling_risk_and_transaction_sensitivity():
    returns, positions, fills = _account()
    report = pf.create_full_tear_sheet(
        returns,
        positions=positions,
        transactions=fills,
        unadjusted_returns=returns,
        estimate_intraday=False,
        return_fig=True,
        periods_per_year=12,
    )
    for title in (
        "Performance statistics",
        "Rolling volatility",
        "Rolling Sharpe",
        "Slippage sensitivity",
    ):
        assert _figure(report, title).layout.meta["periods_per_year"] == 12


def test_full_hidden_account_removes_asset_identity_from_every_selected_export():
    import json

    from plotly.utils import PlotlyJSONEncoder

    returns, positions, fills = _account()
    report = pf.create_full_tear_sheet(
        returns,
        positions=positions,
        transactions=fills,
        sector_mappings={"PRIVATE_A": "PRIVATE_A", "PRIVATE_B": "PRIVATE_B"},
        hide_positions=True,
        estimate_intraday=False,
        return_fig=True,
    )
    assert "PRIVATE_A" not in json.dumps(report.to_dict(), cls=PlotlyJSONEncoder)
    assert "PRIVATE_B" not in json.dumps(report.to_dict(), cls=PlotlyJSONEncoder)


def test_incomplete_factor_inputs_raise_before_a_partial_full_report_is_displayed():
    returns, positions, _ = _account()
    with pytest.raises(ValueError, match="factor_returns.*factor_loadings|both.*factor"):
        pf.create_full_tear_sheet(
            returns,
            positions=positions,
            factor_returns=pd.DataFrame({"market": returns}),
            return_fig=True,
        )


def test_full_capacity_and_supplied_factor_routes_keep_every_table_and_the_selected_scale(tmp_path):
    from test_pyfolio_capacity_sheet import _inputs

    returns, positions, fills, market = _inputs()
    factors = pd.DataFrame({"market": 0.01}, index=returns.index)
    keys = pd.MultiIndex.from_product([returns.index, ["A", "B"]], names=["dt", "ticker"])
    loadings = pd.DataFrame({"market": [2.0, 1] * len(returns)}, index=keys)
    originals = [frame.copy(deep=True) for frame in (market, factors, loadings)]
    report = pf.create_full_tear_sheet(
        returns,
        positions,
        fills,
        market,
        factor_returns=factors,
        factor_loadings=loadings,
        factor_partitions=None,
        estimate_intraday=False,
        periods_per_year=12,
        return_fig=True,
    )
    table = _figure(report, "Whole backtest: liquidation").data[0]
    assert list(table.cells.values[0]) == ["A", "B"]
    assert list(table.cells.values[2]) == ["50", "25"]
    assert sum(name.startswith("Capacity/") for name in report.figures) == 5
    assert sum(name.startswith("Attribution/") for name in report.figures) == 5
    assert (
        _figure(report, "Capital base performance sweep").layout.meta["capacity_report"][
            "periods_per_year"
        ]
        == 12
    )
    np.testing.assert_allclose(
        _figure(report, "Panel 1").data[2].y,
        (1.004 ** np.arange(1, 11)) - 1,
        rtol=1e-12,
        atol=1e-15,
    )
    assert (
        _figure(report, "Attribution statistics").layout.meta["description"].startswith("Supplied")
    )
    report.save_html(tmp_path / "account-with-optional-analytics.html", include_plotlyjs=True)
    for before, after in zip(originals, (market, factors, loadings), strict=True):
        pd.testing.assert_frame_equal(before, after)


def test_full_round_trip_route_keeps_actual_closing_inventory_separate_from_inferred_peak():
    from test_pyfolio_round_trip_report import account

    returns, positions, fills = account()
    with pytest.warns(UserWarning, match="intraday"):
        report = pf.create_full_tear_sheet(
            returns,
            positions,
            fills,
            round_trips=True,
            return_fig=True,
        )
    assert report.round_trips.executions.height == 12
    assert report.round_trips.completed_trades().height == 6
    assert report.round_trips.synthetic_segments is None
    assert report.round_trip_statistics.observations["gross_pnl"].to_list() == [2, -1, 3, -2, 4, -3]
    assert report.availability["composition"]["positions_kind"] == "estimated_intraday"


def test_tied_peak_candidates_do_not_duplicate_the_simple_reports_actual_return_or_cost_observation():
    from test_pyfolio_round_trip_report import _two_day_tied_account

    returns, positions, fills = _two_day_tied_account()
    report = pf.create_simple_tear_sheet(
        returns,
        positions,
        fills,
        slippage=10,
        estimate_intraday=True,
        return_fig=True,
    )
    wealth = _figure(report, "Cumulative wealth").data[0]
    assert len(wealth.y) == 2
    assert wealth.y[-1] == pytest.approx(1.004393082)
    assert len(_figure(report, "Exposure").data[0].y) == 3


def test_actual_reference_extra_slippage_duplicates_a_daily_return_for_tied_exposure_candidates():
    from test_pyfolio_round_trip_report import _two_day_tied_account

    reference = pytest.importorskip("pyfolio.txn")
    returns, positions, fills = _two_day_tied_account()
    selected = pf.utils.estimate_intraday(returns, positions, fills)
    original = reference.adjust_returns_for_slippage(returns, selected, fills, 10)
    assert list(original) == pytest.approx([-0.0004, -0.0004, 0.004795])
    adjusted = pf.txn.adjust_returns_for_slippage(returns, selected, fills, 10)
    assert list(adjusted) == pytest.approx([-0.0004, 0.004795])
    assert (1 + adjusted).prod() == pytest.approx(1.004393082)


def test_conflicting_same_date_nav_cannot_silently_choose_a_slippage_denominator():
    returns, positions, fills = _account()
    duplicate = positions.iloc[[0]].copy(deep=True)
    duplicate["cash"] += 100
    ambiguous = pd.concat([positions.iloc[:1], duplicate, positions.iloc[1:]])
    with pytest.raises(ValueError, match="disagree.*NAV"):
        pf.txn.adjust_returns_for_slippage(returns, ambiguous, fills, 10)


def test_full_hide_identity_also_covers_capacity_tables_attribution_and_actual_round_trip_exports():
    import json

    from plotly.utils import PlotlyJSONEncoder
    from test_pyfolio_round_trip_report import account

    returns, positions, fills = account()
    positions = positions.rename(columns={"A": "PRIVATE_A"})
    fills["symbol"] = "PRIVATE_A"
    market = pd.DataFrame(
        {"PRIVATE_A": [100.0, 1e6] * len(returns)},
        index=pd.MultiIndex.from_product([returns.index, ["price", "volume"]]),
    )
    factors = pd.DataFrame({"market": 0.001}, index=returns.index)
    keys = pd.MultiIndex.from_product(
        [returns.index, ["PRIVATE_A", "PRIVATE_UNUSED"]], names=["dt", "ticker"]
    )
    loadings = pd.DataFrame({"market": [1.0, 2] * len(returns)}, index=keys)
    report = pf.create_full_tear_sheet(
        returns,
        positions,
        fills,
        market,
        factor_returns=factors,
        factor_loadings=loadings,
        factor_partitions=None,
        round_trips=True,
        hide_positions=True,
        sector_mappings={"PRIVATE_A": "PRIVATE_SECTOR"},
        estimate_intraday=False,
        return_fig=True,
    )
    payload = json.dumps(report.to_dict(), cls=PlotlyJSONEncoder)
    assert all(label not in payload for label in ("PRIVATE_A", "PRIVATE_UNUSED", "PRIVATE_SECTOR"))
    assert report.round_trips.executions.height == 12
    assert report.round_trip_statistics.observations["gross_pnl"].sum() == pytest.approx(3)


@pytest.mark.parametrize("kind", ["simple", "full"])
@pytest.mark.parametrize("live", [False, True])
def test_actual_reference_complete_account_workflows_keep_nested_samples_and_option_effects(
    kind, live
):
    from test_pyfolio_returns_report import _inputs

    reference = pytest.importorskip("pyfolio.tears")
    pyplot = pytest.importorskip("matplotlib.pyplot")
    returns, benchmark = _inputs()
    wealth = (1 + returns).cumprod()
    positions = pd.DataFrame({"A": 600 * wealth, "cash": 400 * wealth})
    fills = pd.DataFrame(
        {"symbol": ["A"] * 2, "amount": [1.0, -1], "price": [100.0, 101]},
        index=returns.index[[0, 260]] + pd.Timedelta(hours=15),
    )
    denominator = "portfolio_value" if live else "AGB"
    options = {
        "positions": positions,
        "transactions": fills,
        "benchmark_rets": benchmark,
        "live_start_date": returns.index[260] if live else None,
        "estimate_intraday": False,
        "turnover_denom": denominator,
    }
    try:
        pyplot.close("all")
        getattr(reference, f"create_{kind}_tear_sheet")(
            returns,
            **options,
            **({"set_context": False, "cone_std": None} if kind == "full" else {}),
        )
        originals = [pyplot.figure(number) for number in pyplot.get_fignums()]
        for figure in originals:
            figure.set_dpi(20)
            figure.canvas.draw()
        actual = getattr(pf, f"create_{kind}_tear_sheet")(
            returns,
            **options,
            return_fig=True,
            random_seed=4,
            num_samples=16,
            **({"cone_std": None} if kind == "full" else {}),
        )
        wealth_axis = originals[0].axes[0]
        for i in range(3 if live else 2):
            np.testing.assert_allclose(
                _figure(actual, "Cumulative wealth").data[i].y,
                wealth_axis.lines[i].get_ydata(),
                rtol=1e-12,
                atol=1e-12,
            )
        axes = [axis for figure in originals for axis in figure.axes]
        sharpe = next(axis for axis in axes if axis.get_ylabel() == "Sharpe ratio")
        np.testing.assert_allclose(
            _figure(actual, "Rolling Sharpe").data[0].y,
            sharpe.lines[0].get_ydata(),
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        )
        turnover = next(axis for axis in axes if axis.get_title() == "Daily turnover")
        if kind == "full" and live:
            # Reference full forwards this choice to statistics, but drops it at its transaction sheet.
            assert turnover.lines[0].get_ydata()[0] == pytest.approx(100 / (300 * wealth.iloc[0]))
            assert _figure(actual, "Turnover").data[0].y[0] == pytest.approx(
                100 / (1000 * wealth.iloc[0])
            )
        else:
            np.testing.assert_allclose(
                _figure(actual, "Turnover").data[0].y,
                turnover.lines[0].get_ydata(),
                rtol=1e-12,
            )
        assert len(originals) == (4 if kind == "full" else 1)
    finally:
        pyplot.close("all")
