"""Empirical report distributions must use actual date/bucket observations without RNG changes."""

import json
import math
import re
import shutil
import subprocess

import numpy as np
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
from ml4t.diagnostic.visualization.signal import plot_quantile_returns_violin


def _inputs(nanoseconds=False, one_bucket=False):
    dates = pd.bdate_range("2025-01-02", periods=7, tz="UTC")
    if nanoseconds:
        dates = pd.DatetimeIndex([dates[0] + pd.Timedelta(i * 100, unit="ns") for i in range(7)])
    prices = pd.DataFrame(
        {
            "A": [128.0, 144.0, 180.0, 202.5, 253.125, 284.765625, 355.95703125],
            "B": [128.0, 160.0, 140.0, 175.0, 153.125, 191.40625, 167.48046875],
            "C": [128.0, 192.0, 288.0, 432.0, 648.0, 972.0, 1458.0],
            "D": [128.0, 256.0, 288.0, 576.0, 648.0, 1296.0, 1458.0],
        },
        index=dates,
    )
    factor = (
        prices.pct_change(fill_method=None)
        .iloc[[1, 3]]
        .stack(future_stack=True)
        .rename("factor")
        .rename_axis(["date", "asset"])
        .drop((dates[3], "C"))
    )
    marks = (
        prices.stack(future_stack=True).rename("close").rename_axis(["date", "asset"]).reset_index()
    )
    returns = (
        compute_forward_returns(marks, periods=[1, 2], group_col="asset")
        .set_index(["date", "asset"])[["fwd_ret_1", "fwd_ret_2"]]
        .rename(columns={"fwd_ret_1": "1D", "fwd_ret_2": "2D"})
    )
    panel = prepare_signal_panel(
        factor,
        returns,
        quantiles=1 if one_bucket else 2,
        groups={"A": "growth", "C": "growth", "B": "value", "D": "value"},
        max_loss=0,
    )
    assert panel.data.height == 7
    return factor.reset_index(), marks, panel, dates


def _analysis(route, factor, marks, panel):
    if route == "owned":
        return panel.analyze(min_assets=2)
    if route == "prepared":
        return analyze_prepared_signal(
            panel.data, periods=(1, 2), quantiles=panel.data["quantile"].max(), min_assets=2
        )
    return analyze_signal(
        factor,
        marks,
        periods=(1, 2),
        quantiles=panel.data["quantile"].max(),
        price_col="close",
        filter_zscore=None,
        min_assets=2,
        compute_turnover_flag=False,
    )


def _values(figure):
    return {trace.name: np.asarray(trace.y, dtype=float) for trace in figure.data}


@pytest.mark.parametrize("route", ["raw", "prepared", "owned"])
@pytest.mark.parametrize("nanoseconds", [False, True])
@pytest.mark.parametrize("period", ["1D", "2D"])
def test_quote_to_report_violin_uses_date_bucket_means_not_normal_draws_or_asset_samples(
    route, nanoseconds, period
):
    factor, marks, panel, _ = _inputs(nanoseconds)
    result = _analysis(route, factor, marks, panel)
    figure = plot_quantile_returns_violin(result.to_quantile_result(), period=period)
    expected = (
        {"Q1": [0.0625, 0.0625], "Q2": [0.3125, 0.125]}
        if period == "1D"
        else {"Q1": [0.25, 0.25], "Q2": [1.25, 1.25]}
    )
    actual = _values(figure)
    assert set(actual) == set(expected)
    for label, values in expected.items():
        np.testing.assert_allclose(sorted(actual[label]), sorted(values), rtol=0, atol=1e-14)
    assert result.quantile_returns["1D"][2] == 0.25
    assert figure.layout.meta["sample_unit"] == "date_bucket_mean"
    assert figure.layout.meta["synthetic"] is False


def test_plotting_actual_returns_does_not_reset_caller_random_state():
    factor, marks, panel, _ = _inputs()
    result = _analysis("owned", factor, marks, panel)
    original = np.random.get_state()
    try:
        np.random.seed(191)
        before = np.random.get_state()
        plot_quantile_returns_violin(result.to_quantile_result())
        after = np.random.get_state()
        assert all(np.array_equal(left, right) for left, right in zip(before, after, strict=True))
    finally:
        np.random.set_state(original)


@pytest.mark.parametrize("route", ["raw", "prepared", "owned"])
def test_empirical_observation_keys_counts_and_json_are_owned_and_exact(route, tmp_path):
    factor, marks, panel, dates = _inputs(True)
    result = _analysis(route, factor, marks, panel)
    observations = result.quantile_observations
    assert observations.height == 8
    assert sorted(observations["date"].unique().cast(pl.Int64).to_list()) == [
        day.value for day in dates[[1, 3]]
    ]
    assert observations["sample_unit"].unique().to_list() == ["asset"]
    assert observations["return_adjustment"].unique().to_list() == ["none"]
    one = observations.filter(pl.col("return_column") == "1D").sort("date", "quantile")
    assert one["n_obs"].to_list() == [2, 2, 2, 1]
    assert one["mean"].to_list() == [0.0625, 0.3125, 0.0625, 0.125]
    model = result.to_quantile_result()
    assert model.get_dataframe("observations").equals(observations)
    file = str(tmp_path / "result.json")
    result.to_json(file)
    loaded = SignalResult.from_json(file)
    assert loaded.quantile_observations.equals(observations)
    restored = type(model).model_validate_json(model.model_dump_json())
    assert restored.get_dataframe("observations").equals(observations)
    observations.replace_column(
        observations.get_column_index("mean"), pl.Series("mean", [99.0] * observations.height)
    )
    assert result.quantile_observations["mean"].to_list() != observations["mean"].to_list()


@pytest.mark.parametrize("one_bucket", [False, True])
def test_actual_undefined_quantile_statistics_roundtrip_without_zero_substitution(one_bucket):
    factor, marks, panel, _ = _inputs(one_bucket=one_bucket)
    model = _analysis("owned", factor, marks, panel).to_quantile_result()
    assert np.isnan(model.rank_correlation["1D"])
    restored = type(model).model_validate_json(model.model_dump_json())
    assert np.isnan(restored.rank_correlation["1D"])
    for name in [
        "mean_returns",
        "std_returns",
        "spread_mean",
        "spread_std",
        "spread_t_stat",
        "spread_p_value",
        "spread_ci_lower",
        "spread_ci_upper",
        "rank_correlation",
    ]:
        np.testing.assert_equal(getattr(restored, name), getattr(model, name))
    assert restored.get_dataframe("observations").equals(model.get_dataframe("observations"))


@pytest.mark.parametrize("representation", ["pandas", "polars", "dict"])
def test_explicit_supplied_raw_values_are_not_silently_reduced_to_date_bucket_means(representation):
    factor, marks, panel, _ = _inputs()
    result = _analysis("owned", factor, marks, panel)
    values = result.prepared_data.select("quantile", "1D_fwd_return")
    raw = (
        values.to_pandas()
        if representation == "pandas"
        else values.to_dict(as_series=False)
        if representation == "dict"
        else values
    )
    figure = plot_quantile_returns_violin(result.to_quantile_result(), factor_data=raw, period="1D")
    actual = _values(figure)
    np.testing.assert_array_equal(sorted(actual["Q1"]), [-0.125, -0.125, 0.25, 0.25])
    np.testing.assert_array_equal(sorted(actual["Q2"]), [0.125, 0.125, 0.5])
    assert figure.layout.meta["sample_unit"] == "supplied_value"


def test_legacy_summary_without_empirical_data_does_not_become_a_simulated_distribution():
    factor, marks, panel, _ = _inputs()
    model = _analysis("owned", factor, marks, panel).to_quantile_result()
    payload = model.model_dump(mode="json")
    payload.pop("observations_ipc_base64", None)
    legacy = type(model).model_validate(payload)
    with pytest.raises(ValueError, match="observations|empirical"):
        plot_quantile_returns_violin(legacy)


@pytest.mark.parametrize("one_bucket", [False, True])
def test_one_bucket_and_percentile_limits_keep_real_samples_in_declared_return_units(one_bucket):
    factor, marks, panel, _ = _inputs(one_bucket=one_bucket)
    model = _analysis("owned", factor, marks, panel).to_quantile_result()
    figure = plot_quantile_returns_violin(model, period="1D", ylim_percentiles=(10, 90))
    expected = (
        {"Q1": [0.1875, 1 / 12]} if one_bucket else {"Q1": [0.0625, 0.0625], "Q2": [0.3125, 0.125]}
    )
    for label, values in expected.items():
        np.testing.assert_allclose(
            sorted(_values(figure)[label]), sorted(values), atol=1e-14, rtol=0
        )
    limits = np.percentile([value for values in expected.values() for value in values], [10, 90])
    np.testing.assert_allclose(figure.layout.yaxis.range, limits, rtol=0, atol=1e-14)
    assert figure.layout.meta["return_unit"] == "decimal"


@pytest.mark.parametrize(
    "demeaned,group_adjust", [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("representation", ["pandas", "polars"])
def test_explicit_shared_adjustments_supply_real_selected_date_bucket_values(
    demeaned, group_adjust, representation
):
    factor, marks, panel, _ = _inputs()
    model = _analysis("owned", factor, marks, panel).to_quantile_result()
    observations = panel.quantile_statistics(
        by_date=True, demeaned=demeaned, group_adjust=group_adjust
    )
    data = (
        observations.to_pandas().set_index(["date", "quantile"])
        if representation == "pandas"
        else observations
    )
    figure = plot_quantile_returns_violin(model, observations=data, period="1D")
    expected = (
        {"Q1": [-0.125, -0.0625], "Q2": [0.125, 0.125]}
        if group_adjust
        else {"Q1": [-0.125, -1 / 48], "Q2": [0.125, 1 / 24]}
        if demeaned
        else {"Q1": [0.0625, 0.0625], "Q2": [0.3125, 0.125]}
    )
    for label, values in expected.items():
        np.testing.assert_allclose(
            sorted(_values(figure)[label]), sorted(values), rtol=0, atol=1e-14
        )
    assert (
        observations.equals(data)
        if representation == "polars"
        else observations.equals(pl.from_pandas(data.reset_index()))
    )


@pytest.mark.parametrize("width", [1280, 375])
def test_offline_report_browser_renders_actual_empirical_arrays_with_its_own_runtime(
    width, tmp_path
):
    from ml4t.diagnostic.visualization.signal import SignalDashboard

    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chrome")
    if chrome is None:
        pytest.skip("Installed Chrome/Chromium is required for actual report browser checks")
    factor, marks, panel, _ = _inputs(True)
    report = SignalDashboard().generate(_analysis("owned", factor, marks, panel).to_tear_sheet())
    probe = """<script>
    document.addEventListener('DOMContentLoaded', () => {
        const button = Array.from(document.querySelectorAll('button')).find(node => node.textContent === 'Quantile Analysis');
        button.click();
        const panel = document.getElementById('quantile');
        const marker = document.createElement('output');
        marker.id = 'empirical-browser-check';
        marker.textContent = JSON.stringify({
            runtime: typeof Plotly === 'undefined' ? null : Plotly.version,
            proseFont:getComputedStyle(panel.querySelector('p')).fontFamily,
            proseSize:parseFloat(getComputedStyle(panel.querySelector('p')).fontSize),
            labels:Array.from(panel.querySelectorAll('.gtitle,.xtick text,.ytick text,.legendtext,.annotation-text')).map(node => ({font:getComputedStyle(node).fontFamily,size:parseFloat(getComputedStyle(node).fontSize)})),
            documentWidth:document.documentElement.scrollWidth, viewportWidth:innerWidth,
            traces:Array.from(panel.querySelectorAll('.js-plotly-plot')).flatMap(plot => Array.from(plot._fullData || []).filter(trace => trace.type === 'violin').map(trace => ({name:trace.name,y:Array.from(trace.y || []),customdata:Array.from(trace.customdata || [])})))
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
            "--virtual-time-budget=1500",
            "--dump-dom",
            path.as_uri(),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    match = re.search(r'<output id="empirical-browser-check">([^<]+)</output>', completed.stdout)
    assert match, completed.stderr
    measured = json.loads(match[1])
    assert measured["viewportWidth"] == width
    assert measured["runtime"] is not None
    actual = {trace["name"]: trace for trace in measured["traces"]}
    assert set(actual) == {"Q1", "Q2"}
    np.testing.assert_allclose(actual["Q1"]["y"], [0.0625, 0.0625], rtol=0, atol=1e-14)
    np.testing.assert_allclose(actual["Q2"]["y"], [0.3125, 0.125], rtol=0, atol=1e-14)
    assert [row[1] for row in actual["Q2"]["customdata"]] == [2, 1]
    assert actual["Q2"]["customdata"][0][0].endswith("000000100+00:00")
    assert measured["documentWidth"] <= measured["viewportWidth"]
    assert measured["labels"]
    assert measured["proseSize"] == 16
    assert all(label["font"] == measured["proseFont"] for label in measured["labels"])
    assert all(13 <= label["size"] <= measured["proseSize"] for label in measured["labels"])


@pytest.mark.parametrize("representation", ["pandas", "polars"])
def test_actual_grouped_statistics_require_explicit_selection_not_cross_group_pooling(
    representation,
):
    factor, marks, panel, _ = _inputs()
    model = _analysis("owned", factor, marks, panel).to_quantile_result()
    statistics = panel.quantile_statistics(by_date=True, by_group=True, demeaned=False)
    with pytest.raises(ValueError, match="select one group"):
        plot_quantile_returns_violin(model, observations=statistics)
    selected = statistics.filter(pl.col("group") == "growth")
    before = selected.clone()
    figure = plot_quantile_returns_violin(
        model,
        observations=selected.to_pandas() if representation == "pandas" else selected,
        period="1D",
    )
    np.testing.assert_allclose(_values(figure)["Q1"], [0.25, 0.25], rtol=0, atol=1e-14)
    np.testing.assert_allclose(_values(figure)["Q2"], [0.5], rtol=0, atol=1e-14)
    assert selected.equals(before)


def test_empty_actual_date_query_remains_unavailable_instead_of_generated_samples():
    factor, marks, panel, dates = _inputs()
    model = _analysis("owned", factor, marks, panel).to_quantile_result()
    selected = panel.quantile_statistics(by_date=True).filter(pl.col("date") < dates[0])
    assert selected.is_empty()
    figure = plot_quantile_returns_violin(model, observations=selected, period="1D")
    assert all(len(values) == 0 for values in _values(figure).values())
    assert any(
        "No defined empirical observations" in item.text for item in figure.layout.annotations
    )
