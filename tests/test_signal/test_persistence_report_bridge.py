"""Computed persistence must survive the real signal result, JSON and report workflow."""

import json
import math
import re
import shutil
import subprocess

import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.metrics import compute_forward_returns
from ml4t.diagnostic.signal import (
    SignalResult,
    analyze_prepared_signal,
    analyze_signal,
    prepare_signal_panel,
)


def _inputs(nanoseconds=False, profile="departures"):
    dates = pd.bdate_range("2025-01-02", periods=5, tz="UTC")
    if nanoseconds:
        dates = pd.DatetimeIndex([dates[0] + pd.Timedelta(i * 100, unit="ns") for i in range(5)])
    prices = pd.DataFrame(
        {
            "A": [128.0, 144.0, 180.0, 202.5, 253.125],
            "B": [128.0, 160.0, 140.0, 175.0, 153.125],
            "C": [128.0, 192.0, 288.0, 432.0, 648.0],
            "D": [128.0, 256.0, 288.0, 576.0, 648.0],
        },
        index=dates,
    )
    if profile == "constant":
        prices = pd.concat([prices["A"].rename(asset) for asset in "ABCD"], axis=1)
    elif profile == "singleton":
        prices = prices[["A"]]
    factor = (
        prices.pct_change(fill_method=None)
        .iloc[[1] if profile == "single_date" else [1, 3]]
        .stack(future_stack=True)
        .rename("factor")
        .rename_axis(["date", "asset"])
    )
    if profile == "departures":
        factor = factor.drop((dates[3], "C"))
    marks = (
        prices.stack(future_stack=True).rename("close").rename_axis(["date", "asset"]).reset_index()
    )
    returns = (
        compute_forward_returns(marks, periods=1, group_col="asset")
        .set_index(["date", "asset"])[["fwd_ret_1"]]
        .rename(columns={"fwd_ret_1": "1D"})
    )
    panel = prepare_signal_panel(
        factor, returns, quantiles=1 if profile in {"constant", "singleton"} else 2, max_loss=0
    )
    assert panel.data.height == len(factor)
    return factor.reset_index(), marks, panel, dates


def _analyze(route, factor, marks, panel, enabled=True):
    options = {"compute_turnover_flag": enabled, "autocorrelation_lags": 2, "min_assets": 2}
    quantiles = panel.data["quantile"].max()
    if route == "owned":
        return panel.analyze(**options)
    if route == "prepared":
        return analyze_prepared_signal(panel.data, periods=(1,), quantiles=quantiles, **options)
    return analyze_signal(
        factor,
        marks,
        periods=(1,),
        quantiles=quantiles,
        filter_zscore=None,
        price_col="close",
        **options,
    )


@pytest.mark.parametrize("route", ["raw", "prepared", "owned"])
@pytest.mark.parametrize("nanoseconds", [False, True])
def test_quote_membership_and_full_date_ranks_survive_analysis_and_report(route, nanoseconds):
    factor, marks, panel, dates = _inputs(nanoseconds)
    result = _analyze(route, factor, marks, panel)
    assert result.turnover == {"1D": 0.25}
    assert all(math.isnan(value) for value in result.autocorrelation)
    sheet = result.to_tear_sheet("quote-derived persistence")
    assert sheet.turnover_analysis is not None
    turnover = sheet.turnover_analysis.get_dataframe("dated_turnover")
    rank = sheet.turnover_analysis.get_dataframe("dated_autocorrelation")
    assert turnover.height == 4 and rank.height == 4
    assert turnover["turnover_method"].unique().to_list() == ["entrants_over_current"]
    assert turnover["lag_unit"].unique().to_list() == ["observations"]
    assert rank["rank_scope"].unique().to_list() == ["date"]
    assert rank["correlation_method"].unique().to_list() == ["pearson"]
    assert sorted(pd.Timestamp(value).value for value in turnover["date"].to_pandas().unique()) == [
        day.value for day in dates[[1, 3]]
    ]
    current = turnover.filter(pl.col("date") == pl.col("date").max()).sort("quantile")
    assert current["turnover"].to_list() == [0.0, 0.0]
    assert current["n_current"].to_list() == [2, 1] and current["n_previous"].to_list() == [2, 2]
    defined = rank.filter(pl.col("status") == "defined").to_dicts()
    assert len(defined) == 1 and defined[0]["n_obs"] == 3
    assert defined[0]["autocorrelation"] == pytest.approx(9 / math.sqrt(84))
    assert sheet.turnover_analysis.get_dataframe("summary")["period"].to_list() == ["1obs", "2obs"]


@pytest.mark.parametrize("route", ["raw", "prepared", "owned"])
def test_disabled_persistence_is_not_synthesized_in_the_report(route):
    factor, marks, panel, _ = _inputs()
    result = _analyze(route, factor, marks, panel, enabled=False)
    assert result.to_tear_sheet().turnover_analysis is None
    assert result.to_turnover_result() is None
    assert result.dated_turnover is None and result.dated_autocorrelation is None


@pytest.mark.parametrize("route", ["raw", "prepared", "owned"])
def test_json_and_independent_result_views_preserve_exact_persistence(route, tmp_path):
    factor, marks, panel, _ = _inputs(True)
    factor_before, marks_before, panel_before = (
        factor.copy(deep=True),
        marks.copy(deep=True),
        panel.data,
    )
    result = _analyze(route, factor, marks, panel)
    path = str(tmp_path / "signal.json")
    result.to_json(path)
    loaded = SignalResult.from_json(path)
    assert loaded.dated_turnover.equals(result.dated_turnover)
    assert loaded.dated_autocorrelation.equals(result.dated_autocorrelation)
    native = result.to_turnover_result()
    exported = native.get_dataframe("dated_turnover")
    exported.replace_column(
        exported.get_column_index("turnover"), pl.Series("turnover", [1.0] * exported.height)
    )
    assert native.get_dataframe("dated_turnover").equals(result.dated_turnover)
    roundtrip = type(native).model_validate_json(native.model_dump_json())
    assert roundtrip.get_dataframe("dated_turnover").equals(result.dated_turnover)
    assert roundtrip.get_dataframe("dated_autocorrelation").equals(result.dated_autocorrelation)
    assert (
        loaded.to_tear_sheet()
        .turnover_analysis.get_dataframe("dated_autocorrelation")
        .equals(result.dated_autocorrelation)
    )
    pd.testing.assert_frame_equal(factor, factor_before)
    pd.testing.assert_frame_equal(marks, marks_before)
    assert panel.data.equals(panel_before)


def test_result_to_html_displays_computed_persistence_without_invented_confidence(tmp_path):
    from ml4t.diagnostic.visualization.signal import SignalDashboard

    factor, marks, panel, _ = _inputs()
    result = _analyze("owned", factor, marks, panel)
    html = SignalDashboard().generate(result.to_tear_sheet("computed persistence"))
    assert "Turnover analysis not available" not in html
    assert "Dated quantile turnover" in html and "Dated rank autocorrelation" in html
    assert "entrants_over_current" in html and "observations" in html
    assert "0.981981" in html
    assert "95% CI" not in html.split("Signal Turnover Analysis", 1)[-1]
    assert result.to_turnover_result().get_dataframe("dated_turnover")[
        "n_overlap"
    ].drop_nulls().to_list() == [2, 1]


@pytest.mark.parametrize("profile", ["constant", "singleton", "single_date"])
@pytest.mark.parametrize("route", ["raw", "prepared", "owned"])
def test_actual_unavailable_quote_estimates_remain_missing_through_report(profile, route):
    from ml4t.diagnostic.visualization.signal import SignalDashboard

    factor, marks, panel, _ = _inputs(profile=profile)
    result = _analyze(route, factor, marks, panel)
    analysis = result.to_turnover_result()
    rank = analysis.get_dataframe("dated_autocorrelation")
    assert rank["autocorrelation"].null_count() == rank.height
    summary = analysis.get_dataframe("summary")
    assert summary["mean_autocorrelation"].null_count() == summary.height
    assert summary["half_life"].null_count() == summary.height
    html = SignalDashboard().generate(result.to_tear_sheet())
    assert "Dated rank autocorrelation" in html and "N/A" in html
    if profile == "single_date":
        assert summary["mean_turnover"].null_count() == summary.height
        turnover_card = html.split("<h3>Turnover (", 1)[1].split("</div>\n            </div>", 1)[0]
        assert "N/A" in turnover_card and ">Low<" not in turnover_card


@pytest.mark.parametrize("corruption", ["one_frame", "encoding", "truncated", "private_field"])
def test_persisted_result_cannot_silently_drop_corrupted_dated_observations(corruption, tmp_path):
    factor, marks, panel, _ = _inputs()
    result = _analyze("owned", factor, marks, panel)
    data = json.loads(result.to_json())
    if corruption == "one_frame":
        data.pop("dated_autocorrelation")
    elif corruption == "encoding":
        data["dated_turnover"]["format"] = "unrecognized"
    elif corruption == "truncated":
        data["dated_turnover"]["data"] = data["dated_turnover"]["data"][:8]
    else:
        data["_dated_turnover"] = "unsupported private payload"
    path = tmp_path / "damaged.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        SignalResult.from_json(str(path))


@pytest.mark.parametrize("width", [1280, 375])
def test_real_browser_tab_click_keeps_persistence_and_parent_report_visible(width, tmp_path):
    from ml4t.diagnostic.visualization.signal import SignalDashboard

    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chrome")
    if chrome is None:
        pytest.skip("Installed Chrome/Chromium is required for actual report browser checks")
    factor, marks, panel, _ = _inputs()
    report = SignalDashboard().generate(_analyze("owned", factor, marks, panel).to_tear_sheet())
    probe = """<script>
    document.addEventListener('DOMContentLoaded', () => {
        const button = Array.from(document.querySelectorAll('button')).find(node => node.textContent === 'Turnover');
        button.click();
        const panel = document.getElementById('turnover');
        const prose = panel.querySelector('p');
        const cell = panel.querySelector('td');
        const marker = document.createElement('output');
        marker.id = 'persistence-browser-check';
        marker.textContent = JSON.stringify({
            panelWidth: panel.getBoundingClientRect().width,
            proseWidth: prose.getBoundingClientRect().width,
            cellFont: parseFloat(getComputedStyle(cell).fontSize),
            documentWidth: document.documentElement.scrollWidth,
            viewportWidth: innerWidth
        });
        document.body.appendChild(marker);
    });
    </script>"""
    path = tmp_path / "report.html"
    path.write_text(report.replace("</body>", probe + "</body>"))
    completed = subprocess.run(
        [
            chrome,
            "--headless",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-background-networking",
            "--host-resolver-rules=MAP * ~NOTFOUND",
            f"--user-data-dir={tmp_path / 'browser-profile'}",
            f"--window-size={width},1000",
            "--virtual-time-budget=1000",
            "--dump-dom",
            path.as_uri(),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    match = re.search(r'<output id="persistence-browser-check">([^<]+)</output>', completed.stdout)
    assert match, completed.stderr
    measured = json.loads(match[1])
    assert measured["panelWidth"] > 0 and measured["proseWidth"] > 0
    assert measured["cellFont"] >= 14
    assert measured["documentWidth"] <= measured["viewportWidth"]
