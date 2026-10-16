"""Prepared holdings must reach real portfolio figures, exports and readable HTML."""

import shutil
import subprocess
import sys
from datetime import date

import numpy as np
import pandas as pd
import polars as pl
import pytest
from test_portfolio_positions import _actual_prepared_inputs

from ml4t.diagnostic.evaluation import PortfolioAnalysis
from ml4t.diagnostic.visualization.portfolio import create_portfolio_dashboard


def _prepared(**kwargs):
    returns, positions, valuations, _ = _actual_prepared_inputs()
    analysis = PortfolioAnalysis(returns, positions=positions)
    result = analysis.compute_holdings_analysis(valuations, complete_snapshots=True, **kwargs)
    return analysis, result, valuations


def _traces(figure):
    return {trace.name: list(trace.y) for trace in figure.data}


def test_actual_prepared_holdings_reach_report_with_signed_values_dates_and_exported_inputs():
    analysis, holdings, _ = _prepared()
    report = create_portfolio_dashboard(analysis, holdings=holdings, include_transactions=False)
    assert report.holdings is not holdings
    assert report.availability["positions"]["status"] == "available"
    exposure = report.figures["Holdings Exposure"]
    assert _traces(exposure) == {
        "Long": [0.0, 0.4, 0.4],
        "Short (signed)": [0.0, -0.3, -0.3],
        "Gross": [0.0, 0.7, 0.7],
        "Net": [0.0, 0.1, 0.1],
        "Account cash": [1.0, 0.9, 0.9],
    }
    assert list(exposure.data[0].x) == [
        date.isoformat() for date in pd.DatetimeIndex(analysis.dates.to_pandas())
    ]
    assert exposure.layout.meta["units"] == "fraction of reported NAV"
    assert _traces(report.figures["Position Counts"])["Open"] == [0, 2, 2]
    assert (
        report.figures["Gross-Capital Concentration"].layout.meta["units"]
        == "dimensionless share / HHI"
    )
    assert _traces(report.figures["Signed Position Weights"])["Minimum short"] == [None, -0.3, -0.3]
    top = report.figures["Top Short Holdings"].data[0]
    assert list(top.x) == [-300.0] and list(top.y) == ["B"]
    exported = report.to_dict()
    assert exported["holdings"]["timeline"][-1]["cash"] == 900
    assert exported["holdings"]["positions"][-1]["value"] in {400, -300}
    assert exported["holdings"]["top_positions"]["short"] == [{"asset": "B", "value": -300.0}]
    assert exported["holdings"]["snapshot_kind"] == "observed"


@pytest.mark.parametrize("nav", [None, 0.0, -1000.0])
def test_unusable_nav_reports_raw_exposures_and_reason_instead_of_empty_or_zero_risk(nav):
    analysis, _, valuations = _prepared()
    holdings = analysis.compute_holdings_analysis(
        valuations.with_columns(pl.lit(nav, dtype=pl.Float64).alias("equity")),
        complete_snapshots=True,
    )
    report = create_portfolio_dashboard(analysis, holdings=holdings)
    exposure = report.figures["Holdings Exposure"]
    assert exposure.layout.meta["units"] == "account currency"
    assert _traces(exposure)["Long"] == [0, 400, 400]
    assert _traces(exposure)["Short (signed)"] == [0, -300, -300]
    assert report.availability["holdings_normalization"]["status"] == "unavailable"
    assert "NAV" in report.availability["holdings_normalization"]["reason"]
    assert "Signed Position Weights" not in report.figures
    assert _traces(report.figures["Gross-Capital Concentration"])["Gross HHI"][-1] == pytest.approx(
        25 / 49
    )


def test_partial_nav_retains_normalization_gaps_and_full_raw_account_in_export():
    analysis, _, valuations = _prepared()
    valuations = valuations.with_columns(pl.Series("equity", [1000.0, 1000.0, 0.0]))
    holdings = analysis.compute_holdings_analysis(valuations, complete_snapshots=True)
    report = create_portfolio_dashboard(analysis, holdings=holdings)
    assert _traces(report.figures["Holdings Exposure"])["Gross"] == [0, 0.7, None]
    assert report.figures["Holdings Exposure"].data[0].connectgaps is False
    assert report.to_dict()["holdings"]["timeline"][-1]["gross_exposure"] == 700
    assert report.availability["holdings_normalization"]["status"] == "partial"


def test_missing_cash_is_explicitly_unavailable_not_inferred_from_nav_or_asset_values():
    analysis, _, valuations = _prepared()
    holdings = analysis.compute_holdings_analysis(valuations.drop("cash"), complete_snapshots=True)
    report = create_portfolio_dashboard(analysis, holdings=holdings)
    traces = _traces(report.figures["Holdings Exposure"])
    assert traces["Account cash"] == [None, None, None]
    assert traces["Gross"] == [0, 0.7, 0.7]
    assert report.availability["cash"]["status"] == "unavailable"
    assert report.to_dict()["holdings"]["timeline"][-1]["cash"] is None


@pytest.mark.parametrize("mapping", [{"A": "cash", "B": "cash"}, {"A": "Technology"}])
def test_sector_panels_keep_asset_cash_label_and_unknown_short_separate_from_account_cash(mapping):
    analysis, holdings, _ = _prepared(sector_mapping=mapping)
    report = create_portfolio_dashboard(analysis, holdings=holdings)
    net = report.figures["Sector Net Exposure"]
    gross = report.figures["Sector Gross Exposure"]
    if mapping.get("B") == "cash":
        assert list(net.data[0].y) == [0, 0.1, 0.1]
        assert list(gross.data[0].y) == [0, 0.7, 0.7]
    else:
        (unknown,) = [trace for trace in net.data if trace.meta["sector"] is None]
        assert list(unknown.y) == [0, -0.3, -0.3]
    assert _traces(report.figures["Holdings Exposure"])["Account cash"] == [1.0, 0.9, 0.9]


def test_dated_sector_plot_preserves_missing_group_cells_without_filling_future_labels():
    analysis, _, valuations = _prepared()
    dates = analysis.dates.to_list()
    mapping = pd.DataFrame(
        {
            "date": [dates[-1], *dates],
            "asset": ["A", "B", "B", "B"],
            "sector": ["Tech", "Finance", "Finance", "Finance"],
        }
    )
    holdings = analysis.compute_holdings_analysis(
        valuations, complete_snapshots=True, sector_mapping=mapping
    )
    report = create_portfolio_dashboard(analysis, holdings=holdings)
    (tech,) = [
        trace
        for trace in report.figures["Sector Net Exposure"].data
        if trace.meta["sector"] == "Tech"
    ]
    assert list(tech.y) == [None, None, 0.4]
    assert list(tech.x) == [date.isoformat() for date in dates]


def test_integer_asset_and_numeric_sector_ids_remain_typed_in_chart_payload_and_export():
    returns, positions, valuations, _ = _actual_prepared_inputs()
    long = positions.reset_index().melt(id_vars="date", var_name="asset", value_name="value")
    long["asset"] = long["asset"].map({"A": 1, "B": 2})
    analysis = PortfolioAnalysis(returns, positions=long)
    holdings = analysis.compute_holdings_analysis(
        valuations, complete_snapshots=True, sector_mapping={1: 10, 2: None}
    )
    report = create_portfolio_dashboard(analysis, holdings=holdings)
    assert list(report.figures["Top Short Holdings"].data[0].y) == [2]
    assert report.to_dict()["holdings"]["top_positions"]["short"] == [{"asset": 2, "value": -300.0}]
    assert {trace.meta["sector"] for trace in report.figures["Sector Net Exposure"].data} == {
        10,
        None,
    }


def test_distinct_nanosecond_observation_keys_cannot_alias_through_python_datetime_conversion():
    analysis, holdings, _ = _prepared()
    shifted = pd.DatetimeIndex(analysis.dates.to_pandas()) + pd.Timedelta(1, unit="ns")
    other = PortfolioAnalysis(pd.Series([0.0] * 3, index=shifted))
    with pytest.raises(ValueError, match="observation|keys"):
        create_portfolio_dashboard(other, holdings=holdings)


def test_valid_nanosecond_keys_survive_holdings_chart_transport_and_export_without_truncation():
    returns, positions, valuations, _ = _actual_prepared_inputs()
    dates = returns.index + pd.to_timedelta([1, 2, 3], unit="ns")
    positions.index = dates
    analysis = PortfolioAnalysis(pd.Series([0.0] * 3, index=dates), positions=positions)
    holdings = analysis.compute_holdings_analysis(
        valuations.with_columns(pl.Series("date", dates)), complete_snapshots=True
    )
    report = create_portfolio_dashboard(analysis, holdings=holdings)
    expected = [date.isoformat() for date in dates]
    assert list(report.figures["Holdings Exposure"].data[0].x) == expected
    assert [row["date"] for row in report.to_dict()["holdings"]["timeline"]] == expected


def test_date_labels_are_not_exported_as_invented_midnight_event_timestamps():
    days = pl.Series("date", [date(2025, 1, day) for day in [2, 3, 6]])
    analysis = PortfolioAnalysis(
        [0.0, 0.0, 0.0],
        dates=days,
        positions=pl.DataFrame(
            {"date": days, "A": [0.0, 400.0, 400.0], "B": [0.0, -300.0, -300.0]}
        ),
    )
    holdings = analysis.compute_holdings_analysis(
        pl.DataFrame({"date": days, "equity": [1000.0] * 3, "cash": [1000.0, 900.0, 900.0]}),
        complete_snapshots=True,
    )
    report = create_portfolio_dashboard(analysis, holdings=holdings)
    expected = ["2025-01-02", "2025-01-03", "2025-01-06"]
    assert list(report.figures["Holdings Exposure"].data[0].x) == expected
    assert [row["date"] for row in report.to_dict()["holdings"]["timeline"]] == expected


def test_date_label_holdings_cannot_alias_midnight_timestamps_in_another_valid_analysis():
    days = pl.Series("date", [date(2025, 1, day) for day in [2, 3, 6]])
    analysis = PortfolioAnalysis(
        [0.0, 0.0, 0.0], dates=days, positions=pl.DataFrame({"date": days, "A": [100.0] * 3})
    )
    holdings = analysis.compute_holdings_analysis(
        pl.DataFrame({"date": days, "equity": [1000.0] * 3, "cash": [900.0] * 3}),
        complete_snapshots=True,
    )
    other = PortfolioAnalysis(
        pd.Series([0.0] * 3, index=pd.DatetimeIndex(["2025-01-02", "2025-01-03", "2025-01-06"]))
    )
    with pytest.raises(ValueError, match="observation|keys|Date"):
        create_portfolio_dashboard(other, holdings=holdings)


def test_requested_but_unprepared_positions_have_actionable_reason_and_are_not_silently_ignored():
    analysis, _, _ = _prepared()
    report = create_portfolio_dashboard(analysis, include_positions=True)
    assert report.holdings is None
    assert report.availability["positions"]["status"] == "unavailable"
    assert "complete_snapshots" in report.availability["positions"]["reason"]
    assert report.to_dict()["holdings"] is None


def test_disabled_positions_remain_distinct_from_unavailable_requested_analysis():
    analysis, holdings, _ = _prepared()
    report = create_portfolio_dashboard(analysis, holdings=holdings, include_positions=False)
    assert report.holdings is None and report.availability["positions"]["status"] == "not_requested"
    assert "Holdings Exposure" not in report.figures


def test_mismatched_real_analysis_windows_raise_before_modifying_caller_holdings():
    analysis, holdings, _ = _prepared()
    before = holdings.timeline.clone()
    other = PortfolioAnalysis(
        pd.Series([0.0, 0.0], index=pd.DatetimeIndex(analysis.dates.to_list()[1:]))
    )
    with pytest.raises(ValueError, match="observation|keys"):
        create_portfolio_dashboard(other, holdings=holdings)
    assert holdings.timeline.equals(before)


def test_same_instants_in_a_different_declared_timezone_do_not_silently_match_report_keys():
    analysis, holdings, _ = _prepared()
    dates = pd.DatetimeIndex(analysis.dates.to_pandas())
    other_dates = (
        dates.tz_localize("UTC") if dates.tz is None else dates.tz_convert("America/New_York")
    )
    other = PortfolioAnalysis(pd.Series([0.0] * 3, index=other_dates))
    with pytest.raises(ValueError, match="timezone"):
        create_portfolio_dashboard(other, holdings=holdings)


def test_cash_only_report_has_known_zero_exposure_and_explicit_no_capital_or_side_results():
    analysis, _, valuations = _prepared()
    analysis = PortfolioAnalysis(
        pd.Series([0.0] * 3, index=pd.DatetimeIndex(analysis.dates.to_list())),
        positions=pd.DataFrame(
            {"cash": [1000.0] * 3}, index=pd.DatetimeIndex(analysis.dates.to_list())
        ),
    )
    holdings = analysis.compute_holdings_analysis(
        valuations.with_columns(pl.lit(1000.0).alias("cash")),
        complete_snapshots=True,
        sector_mapping={},
    )
    report = create_portfolio_dashboard(analysis, holdings=holdings)
    assert _traces(report.figures["Holdings Exposure"])["Gross"] == [0, 0, 0]
    assert "Gross-Capital Concentration" not in report.figures
    assert "capital" in report.availability["concentration"]["reason"]
    assert "No" in report.availability["top_long"]["reason"]
    assert "No" in report.availability["sectors"]["reason"]


def test_report_and_exports_own_the_prepared_snapshot_without_changing_analysis():
    analysis, holdings, _ = _prepared(sector_mapping={"A": "Tech"})
    original = analysis.compute_summary_stats().to_dict()
    report = create_portfolio_dashboard(analysis, holdings=holdings)
    holdings.positions.replace_column(2, pl.Series("value", [99.0] * 6))
    assert report.holdings.positions["value"].max() == 400
    export = report.to_dict()
    export["availability"]["positions"]["status"] = "corrupted"
    export["holdings"]["positions"][0]["value"] = 888
    assert report.availability["positions"]["status"] == "available"
    assert report.holdings.positions["value"][0] == 0
    for key, value in original.items():
        np.testing.assert_equal(analysis.compute_summary_stats().to_dict()[key], value)


def test_real_write_interruption_cannot_replace_verified_html_with_a_partial_report(tmp_path):
    pytest.importorskip(
        "resource", reason="POSIX file-size limit required for real write interruption"
    )
    path = tmp_path / "report.html"
    path.write_bytes(b"existing verified report")
    probe = """
import errno
import resource
import signal
import sys
import pandas as pd
from ml4t.diagnostic.evaluation import PortfolioAnalysis
from ml4t.diagnostic.visualization.portfolio import create_portfolio_dashboard
analysis = PortfolioAnalysis(pd.Series([0., .01, -.01],index=pd.date_range('2025-01-01',periods=3)))
report = create_portfolio_dashboard(analysis,include_positions=False,include_transactions=False)
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
resource.setrlimit(resource.RLIMIT_FSIZE,(4096,4096))
try:
    report.save_html(sys.argv[1],include_plotlyjs=True)
except OSError as error:
    assert error.errno == errno.EFBIG, repr(error)
    print('write_interrupted')
else:
    raise AssertionError('Real file-size limit did not interrupt report output')
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe, str(path)], capture_output=True, text=True, timeout=60
    )
    assert completed.returncode == 0, completed.stderr
    assert "write_interrupted" in completed.stdout
    assert path.read_bytes() == b"existing verified report"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("width", [1280, 375])
@pytest.mark.parametrize("positions", [False, True])
def test_real_export_uses_true_viewport_and_readable_leaf_text_for_returns_and_holdings(
    width, positions, tmp_path
):
    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chrome")
    if chrome is None:
        pytest.skip("Installed Chrome/Chromium required for actual report checks")
    playwright = pytest.importorskip("playwright.sync_api")
    analysis, holdings, _ = _prepared()
    options = {"holdings": holdings} if positions else {}
    report = create_portfolio_dashboard(analysis, include_positions=positions, **options)
    path = tmp_path / "portfolio.html"
    report.save_html(path, include_plotlyjs=True)
    errors, requests = [], []
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
        try:
            context = browser.new_context(
                viewport={"width": width, "height": 1000},
                is_mobile=width == 375,
                has_touch=width == 375,
            )
            context.set_offline(True)
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("requestfailed", lambda request: requests.append(request.url))
            page.goto(path.as_uri(), wait_until="load")
            page.wait_for_function(
                "Array.from(document.querySelectorAll('.js-plotly-plot')).every(p=>p._fullLayout)"
            )
            actual = page.evaluate(
                """() => {
                    const roles = '.xtick text,.ytick text,.legendtext,.gtitle,.xtitle,.ytitle,.colorbar text';
                    return {
                        width: innerWidth, scroll: document.documentElement.scrollWidth,
                        body: parseFloat(getComputedStyle(document.body).fontSize),
                        plots: Array.from(document.querySelectorAll('.js-plotly-plot')).map(p => {
                            const b = p.getBoundingClientRect();
                            const labels = Array.from(p.querySelectorAll(roles)).filter(t => t.getBoundingClientRect().width > 0);
                            const axis = p.querySelector('.xtitle')?.getBoundingClientRect();
                            const overlaps = axis && Array.from(p.querySelectorAll('.legendtext')).some(t => {
                                const r = t.getBoundingClientRect();
                                return Math.min(axis.right,r.right)-Math.max(axis.left,r.left)>1 && Math.min(axis.bottom,r.bottom)-Math.max(axis.top,r.top)>1;
                            });
                            return {
                                name: p.closest('.plot-container').querySelector('h2').textContent,
                                width: p._fullLayout.width, visible: b.width>0 && b.height>0,
                                prepared: p.layout.meta?.analysis==='prepared_holdings',
                                text: labels.map(t => parseFloat(getComputedStyle(t).fontSize)),
                                clipped: labels.some(t => {
                                    const r=t.getBoundingClientRect();
                                    return r.left<b.left-1 || r.right>b.right+1 || r.top<b.top-1 || r.bottom>b.bottom+1;
                                }),
                                overlap: Boolean(overlaps),
                                traces: p.layout.meta?.analysis==='prepared_holdings' ? p._fullData.map(t=>({name:t.name,x:Array.from(t.x||[]),y:Array.from(t.y||[])})) : []
                            };
                        })
                    };
                }"""
            )
            assert actual["width"] == width and actual["scroll"] <= width
            assert len(actual["plots"]) == (15 if positions else 8)
            assert all(plot["visible"] for plot in actual["plots"])
            assert all(plot["width"] <= width for plot in actual["plots"])
            assert all(
                14 <= size <= actual["body"] for plot in actual["plots"] for size in plot["text"]
            )
            assert all(
                not plot["clipped"] and not plot["overlap"]
                for plot in actual["plots"]
                if plot["prepared"]
            )
            if positions:
                exposure = next(
                    plot for plot in actual["plots"] if plot["name"] == "Holdings Exposure"
                )
                assert next(
                    trace["y"] for trace in exposure["traces"] if trace["name"] == "Gross"
                ) == [0, 0.7, 0.7]
                assert next(
                    trace["y"] for trace in exposure["traces"] if trace["name"] == "Short (signed)"
                ) == [0, -0.3, -0.3]
        finally:
            browser.close()
    assert not errors and not requests
