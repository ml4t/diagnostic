"""Complete capacity reports must preserve thresholds, samples and model units."""

import os
import shutil
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.compat.pyfolio import tears

pytest.importorskip("plotly")


class _Tables(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.tables = {}
        self.current = None
        self.row = []
        self.cell = None
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.current = dict(attrs).get("id", f"table-{len(self.tables)}")
            if self.current is not None:
                self.tables[self.current] = []
        elif self.current is not None and tag == "tr":
            self.row = []
        elif self.current is not None and tag in {"th", "td"}:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag == "table":
            self.current = None
        elif self.current is not None and tag in {"th", "td"} and self.cell is not None:
            self.row.append("".join(self.cell).strip())
            self.cell = None
        elif self.current is not None and tag == "tr":
            self.tables[self.current].append(self.row)


def _inputs():
    dates = pd.date_range("2025-01-13", periods=10, freq="B", tz="UTC")
    returns = pd.Series(
        [0.01, -0.01, 0.02, -0.015, 0.005, -0.005, 0.012, -0.004, 0.006, -0.003], index=dates
    )
    positions = pd.DataFrame(
        {"A": [100.0] * 10, "B": [200.0] * 10, "cash": [700.0] * 10}, index=dates
    )
    transactions = pd.DataFrame(
        {"symbol": ["A"], "amount": [60.0], "price": [10.0]},
        index=dates[7:8] + pd.Timedelta(hours=15),
    )
    market = pd.DataFrame(
        {"A": [10.0, 1000.0] * 10, "B": [20.0, 2000.0] * 10},
        index=pd.MultiIndex.from_product([dates, ["price", "volume"]]),
    )
    return returns, positions, transactions, market


def _rows(html, name):
    rows = _Tables(html).tables["capacity-" + name]
    return [row for row in rows if row and row[0] in {"A", "B"}]


def test_complete_capacity_sheet_renders_independent_liquidation_and_participation_tables():
    inputs = _inputs()
    copies = [value.copy(deep=True) for value in inputs]
    html = tears.create_capacity_tear_sheet(*inputs, estimate_intraday=False)
    full = _rows(html, "liquidation-full")
    assert {row[0]: float(row[2]) for row in full} == {"A": 50.0, "B": 25.0}
    assert {row[0]: float(row[3]) for row in full} == {"A": 10.0, "B": 20.0}
    assert {row[0]: float(row[2]) for row in _rows(html, "participation-full")} == {"A": 6.0}
    assert _rows(html, "liquidation-recent") == full
    assert "Sharpe" in html and "dimensionless" in html
    for original, copy in zip(inputs, copies, strict=True):
        if isinstance(original, pd.Series):
            pd.testing.assert_series_equal(original, copy)
        else:
            pd.testing.assert_frame_equal(original, copy)


def test_caller_liquidation_threshold_applies_to_both_full_and_recent_views():
    html = tears.create_capacity_tear_sheet(
        *_inputs(),
        liquidation_daily_vol_limit=0.1,
        trade_daily_vol_limit=0.1,
        last_n_days=2,
        days_to_liquidate_limit=75,
        estimate_intraday=False,
    )
    assert {row[0]: float(row[2]) for row in _rows(html, "liquidation-full")} == {"A": 100.0}
    assert {row[0]: float(row[2]) for row in _rows(html, "liquidation-recent")} == {"A": 100.0}
    assert _rows(html, "participation-full") == []
    assert "10%" in html and "75" in html


def test_capacity_sheet_propagates_explicit_annualization_to_the_actual_curve():
    returns, positions, transactions, market = _inputs()
    market.loc[market.index.get_level_values(1) == "volume", :] *= 1e6
    monthly = tears.create_capacity_tear_sheet(
        returns,
        positions,
        transactions,
        market,
        estimate_intraday=False,
        return_fig=True,
        periods_per_year=12,
    )
    daily = tears.create_capacity_tear_sheet(
        returns,
        positions,
        transactions,
        market,
        estimate_intraday=False,
        return_fig=True,
        periods_per_year=252,
    )
    assert monthly.data[0].x[0] == daily.data[0].x[0] == 0.1
    assert daily.data[0].y[0] == pytest.approx(monthly.data[0].y[0] * np.sqrt(21))


def test_default_infer_and_forced_tied_estimates_keep_original_candidate_rows(tmp_path):
    returns, positions, _, market = _inputs()
    positions.loc[:, ["A", "B"]] = 0.0
    positions["cash"] = 1000.0
    instants = positions.index[7] + pd.to_timedelta([12, 14, 16], unit="h")
    transactions = pd.DataFrame(
        {"symbol": ["A"] * 3, "amount": [2.0, -4.0, 2.0], "price": [10.0] * 3}, index=instants
    )
    for estimate in ("infer", True):
        output = tmp_path / f"capacity-{estimate}.html"
        figure = tears.create_capacity_tear_sheet(
            returns,
            positions,
            transactions,
            market,
            estimate_intraday=estimate,
            return_fig=True,
            days_to_liquidate_limit=0,
            output_file=output,
        )
        profile = figure.layout.meta["capacity_report"]
        assert profile["positions_kind"] == "estimated_intraday"
        assert profile["tied_date_observations"] == 1
        assert profile["position_observations"] == 11
        rows = _rows(output.read_text(encoding="utf-8"), "liquidation-full")
        assert {row[0]: float(row[2]) for row in rows} == {"A": 10.0}
        assert {row[0]: float(row[3]) for row in rows} == {"A": 2.0}


def test_invalid_capacity_option_preserves_the_previous_output(tmp_path):
    output = tmp_path / "capacity.html"
    output.write_text("previous report", encoding="utf-8")
    with pytest.raises(ValueError, match="liquidation_daily_vol_limit"):
        tears.create_capacity_tear_sheet(
            *_inputs(), liquidation_daily_vol_limit=1.1, output_file=output
        )
    assert output.read_text(encoding="utf-8") == "previous report"


def test_capacity_figure_mode_still_writes_all_tables_and_offline_html(tmp_path):
    output = tmp_path / "capacity.html"
    figure = tears.create_capacity_tear_sheet(
        *_inputs(), estimate_intraday=False, return_fig=True, output_file=output
    )
    html = output.read_text(encoding="utf-8")
    assert len(_rows(html, "liquidation-full")) == 2
    assert len(figure.data) == 1
    assert 'src="https://cdn.plot.ly' not in html


def test_all_cash_empty_execution_sheet_preserves_returns_without_fabricated_activity(tmp_path):
    returns, positions, transactions, market = _inputs()
    positions.loc[:, ["A", "B"]] = 0.0
    positions["cash"] = 1000.0
    output = tmp_path / "empty.html"
    figure = tears.create_capacity_tear_sheet(
        returns, positions, transactions.iloc[:0], market, output_file=output, return_fig=True
    )
    html = output.read_text(encoding="utf-8")
    assert all(
        _rows(html, name) == []
        for name in [
            "liquidation-full",
            "liquidation-recent",
            "participation-full",
            "participation-recent",
        ]
    )
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    np.testing.assert_allclose(figure.data[0].y, expected)
    assert figure.layout.meta["capacity_report"]["positions_kind"] == "supplied"


def test_unknown_active_volume_is_visible_and_never_becomes_free_capacity(tmp_path):
    returns, positions, transactions, market = _inputs()
    market.loc[(returns.index[7], "volume"), "A"] = np.nan
    output = tmp_path / "unknown.html"
    figure = tears.create_capacity_tear_sheet(
        returns,
        positions,
        transactions,
        market,
        estimate_intraday=False,
        output_file=output,
        return_fig=True,
    )
    rows = _rows(output.read_text(encoding="utf-8"), "participation-full")
    assert len(rows) == 1 and rows[0][0] == "A" and rows[0][2] == "Unavailable"
    assert set(figure.data[0].y) == {None}
    assert any("Unavailable" in annotation.text for annotation in figure.layout.annotations)


def test_unknown_initial_return_disables_sweep_without_erasing_liquidation_tables(tmp_path):
    returns, positions, transactions, market = _inputs()
    returns.iloc[0] = np.nan
    output = tmp_path / "no-capital.html"
    figure = tears.create_capacity_tear_sheet(
        returns,
        positions,
        transactions,
        market,
        estimate_intraday=False,
        output_file=output,
        return_fig=True,
    )
    assert len(figure.data) == 0
    assert figure.layout.meta["capacity_report"]["beginning_capital"] is None
    assert {
        row[0]: float(row[2])
        for row in _rows(output.read_text(encoding="utf-8"), "liquidation-full")
    } == {"A": 50.0, "B": 25.0}


def test_actual_published_full_capacity_sheet_matches_the_defined_sample_and_curve(monkeypatch):
    reference = pytest.importorskip("pyfolio.tears")
    import matplotlib.pyplot as plt

    inputs = _inputs()
    inputs[2]["amount"] = 0.006
    displayed = []
    monkeypatch.setattr(reference.utils, "display", lambda value: displayed.append(value.data))
    original = reference.create_capacity_tear_sheet(
        *[value.copy(deep=True) for value in inputs], estimate_intraday=False, return_fig=True
    )
    try:
        assert len(displayed) == 4
        candidate = tears.create_capacity_tear_sheet(
            *inputs, estimate_intraday=False, return_fig=True
        )
        curve = original.axes[0].lines[0]
        np.testing.assert_allclose(candidate.data[0].x, curve.get_xdata())
        np.testing.assert_allclose(candidate.data[0].y, curve.get_ydata(), rtol=1e-12)
        html = tears.create_capacity_tear_sheet(*inputs, estimate_intraday=False)
        for legacy, name in zip(
            displayed,
            [
                "liquidation-full",
                "liquidation-recent",
                "participation-full",
                "participation-recent",
            ],
            strict=True,
        ):
            original_rows = next(iter(_Tables(legacy).tables.values()))
            rows = [row for row in original_rows if row and row[0] in {"A", "B"}]
            actual = _rows(html, name)
            assert len(actual) == len(rows)
            for new, old in zip(actual, rows, strict=True):
                assert new[:2] == old[:2]
                np.testing.assert_allclose(
                    [float(value) for value in new[2:]], [float(value) for value in old[2:]]
                )
    finally:
        plt.close(original)


def test_actual_reference_recent_threshold_bug_is_not_ported_into_the_report(monkeypatch):
    reference = pytest.importorskip("pyfolio.tears")
    import matplotlib.pyplot as plt

    inputs = _inputs()
    inputs[2]["amount"] = 0.006
    displayed = []
    monkeypatch.setattr(reference.utils, "display", lambda value: displayed.append(value.data))
    original = reference.create_capacity_tear_sheet(
        *[value.copy(deep=True) for value in inputs],
        estimate_intraday=False,
        days_to_liquidate_limit=75,
        return_fig=True,
    )
    try:
        full, recent = [_Tables(html).tables["table-0"] for html in displayed[:2]]
        assert [row for row in full if row and row[0] in {"A", "B"}] == []
        assert [row[0] for row in recent if row and row[0] in {"A", "B"}] == ["A", "B"]
        html = tears.create_capacity_tear_sheet(
            *inputs, estimate_intraday=False, days_to_liquidate_limit=75
        )
        assert _rows(html, "liquidation-full") == _rows(html, "liquidation-recent") == []
    finally:
        plt.close(original)


@pytest.mark.parametrize("width", [375, 1280])
def test_complete_capacity_report_renders_readable_tables_and_actual_curve_offline(tmp_path, width):
    browser_api = pytest.importorskip("playwright.sync_api")
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if chrome is None:
        pytest.skip("Installed Chrome/Chromium is required for actual report rendering")
    returns, positions, transactions, market = _inputs()
    transactions["amount"] = 0.006
    output = tmp_path / "capacity.html"
    figure = tears.create_capacity_tear_sheet(
        returns,
        positions,
        transactions,
        market,
        estimate_intraday=False,
        output_file=output,
        return_fig=True,
    )
    with browser_api.sync_playwright() as runtime:
        browser = runtime.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
        try:
            page = browser.new_page(viewport={"width": width, "height": 1000})
            page.context.set_offline(True)
            errors, failed = [], []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("requestfailed", lambda request: failed.append(request.url))
            page.goto(output.as_uri(), wait_until="load")
            page.wait_for_function("document.querySelector('.js-plotly-plot')._fullLayout")
            state = page.evaluate("""() => {
                const plot=document.querySelector('.js-plotly-plot');
                const body=parseFloat(getComputedStyle(document.body).fontSize);
                return {width:innerWidth,scroll:document.documentElement.scrollWidth,body,
                    tables:[...document.querySelectorAll('.capacity-table')].map(t=>({
                        width:t.getBoundingClientRect().width,parentWidth:t.parentElement.clientWidth,
                        parentScroll:t.parentElement.scrollWidth,
                        sizes:[...t.querySelectorAll('td,th')].map(n=>parseFloat(getComputedStyle(n).fontSize))})),
                    curve:{x:Array.from(plot.data[0].x),y:Array.from(plot.data[0].y)},
                    text:[...plot.querySelectorAll('text')].filter(n=>n.getBoundingClientRect().width>0).map(n=>{
                        const r=n.getBoundingClientRect(),b=plot.getBoundingClientRect();
                        return {text:n.textContent,size:parseFloat(getComputedStyle(n).fontSize),
                            clipped:r.left<b.left-1||r.right>b.right+1||r.top<b.top-1||r.bottom>b.bottom+1};})};
            }""")
            assert not errors and not failed
            assert state["width"] == width and state["scroll"] <= width
            assert len(state["tables"]) == 4
            assert all(
                14 <= size <= state["body"] for table in state["tables"] for size in table["sizes"]
            )
            assert all(
                table["parentWidth"] <= width and table["parentScroll"] >= table["width"] - 1
                for table in state["tables"]
            )
            assert all(
                14 <= leaf["size"] <= state["body"] and not leaf["clipped"]
                for leaf in state["text"]
            ), [
                leaf
                for leaf in state["text"]
                if leaf["clipped"] or not 14 <= leaf["size"] <= state["body"]
            ]
            np.testing.assert_allclose(state["curve"]["x"], figure.data[0].x)
            np.testing.assert_allclose(state["curve"]["y"], figure.data[0].y)
            if destination := os.environ.get("ML4T_CAPACITY_SCREENSHOTS"):
                directory = Path(destination)
                directory.mkdir(parents=True, exist_ok=True)
                page.screenshot(
                    path=str(directory / f"familiar-capacity-{width}.png"), full_page=True
                )
        finally:
            browser.close()
