"""A prepared-price return report must retain samples, scale and missing uncertainty."""

from fractions import Fraction

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml4t.diagnostic.compat.alphalens import utils
from ml4t.diagnostic.signal import PreparedSignalPanel


def _panel(*, singleton=False):
    dates = pd.bdate_range("2025-01-02", periods=6, tz="UTC")
    growth = np.array([[1 / 16, 1 / 8], [0, 1 / 16], [-1 / 16, -1 / 32], [0, 1 / 16], [1 / 16, 0]])
    prices = pd.DataFrame(
        np.vstack([np.full(2, 100.0), 100 * np.cumprod(1 + growth, axis=0)]),
        index=dates,
        columns=["A", "B"],
    )
    factor = pd.Series(
        [-1.0, 1.0] * (1 if singleton else 3),
        index=pd.MultiIndex.from_product(
            [dates[: 1 if singleton else 3], ["A", "B"]], names=["date", "asset"]
        ),
    )
    clean = utils.get_clean_factor_and_forward_returns(
        factor, prices, periods=(1, 3), quantiles=2, max_loss=0, filter_zscore=None, calendar=None
    )
    return PreparedSignalPanel.from_prepared(clean, return_columns=["1D", "3D"], calendar=None)


def test_single_observation_quantile_bar_cannot_present_unknown_error_as_zero():
    pytest.importorskip("plotly")
    from ml4t.diagnostic.visualization.signal import plot_quantile_returns_bar

    result = _panel(singleton=True).analyze(min_assets=2).to_quantile_result()
    assert all(np.isnan(value) for value in result.std_returns["1D"].values())
    figure = plot_quantile_returns_bar(result, period="1D", show_error_bars=True)
    assert list(figure.data[0].error_y.array) == [None, None]
    assert not figure.data[0].error_y.visible


def test_mixed_quantile_bar_draws_only_defined_errors_without_losing_unknown_buckets():
    pytest.importorskip("plotly")
    from ml4t.diagnostic import analyze_prepared_signal
    from ml4t.diagnostic.visualization.signal import plot_quantile_returns_bar

    dates = pd.bdate_range("2025-01-02", periods=2, tz="UTC")
    prices = pd.DataFrame(
        {"A": [100.0, 110.0], "B": [100.0, 120.0], "C": [100.0, 130.0], "D": [100.0, 140.0]},
        index=dates,
    )
    factor = pd.Series(
        [-2.0, -1.0, 1.0, 2.0],
        index=pd.MultiIndex.from_product([dates[:1], list("ABCD")], names=["date", "asset"]),
    )
    clean = utils.get_clean_factor_and_forward_returns(
        factor, prices, periods=(1,), quantiles=3, max_loss=0, filter_zscore=None
    )
    result = analyze_prepared_signal(
        clean, return_columns=["1D"], min_assets=2
    ).to_quantile_result()
    figure = plot_quantile_returns_bar(result, show_error_bars=True)
    assert list(figure.data[0].x) == ["Q1", "Q2", "Q3"]
    assert len(figure.data) == 2 and not figure.data[0].error_y.visible
    assert list(figure.data[1].x) == ["Q1"]
    assert list(figure.data[1].error_y.array) == pytest.approx([0.05])
    assert figure.data[1].legendgroup == figure.data[0].legendgroup
    assert figure.layout.meta["defined_error_count"] == 1
    assert figure.layout.meta["missing_error_count"] == 2


def test_known_zero_quantile_errors_remain_defined_rather_than_hidden_as_missing():
    pytest.importorskip("plotly")
    from ml4t.diagnostic import analyze_prepared_signal
    from ml4t.diagnostic.visualization.signal import plot_quantile_returns_bar

    dates = pd.bdate_range("2025-01-02", periods=2, tz="UTC")
    prices = pd.DataFrame(
        {"A": [100.0, 110.0], "B": [100.0, 110.0], "C": [100.0, 120.0], "D": [100.0, 120.0]},
        index=dates,
    )
    factor = pd.Series(
        [-2.0, -1.0, 1.0, 2.0],
        index=pd.MultiIndex.from_product([dates[:1], list("ABCD")], names=["date", "asset"]),
    )
    clean = utils.get_clean_factor_and_forward_returns(
        factor, prices, periods=(1,), quantiles=2, max_loss=0, filter_zscore=None
    )
    result = analyze_prepared_signal(
        clean, return_columns=["1D"], min_assets=2
    ).to_quantile_result()
    figure = plot_quantile_returns_bar(result, show_error_bars=True)
    assert len(figure.data) == 1 and figure.data[0].error_y.visible
    assert list(figure.data[0].error_y.array) == [0, 0]
    assert figure.layout.meta["defined_error_count"] == 2
    assert figure.layout.meta["missing_error_count"] == 0


def test_price_to_owned_return_report_preserves_exact_weights_products_and_annual_scale():
    from ml4t.diagnostic.signal import analyze_signal_returns

    panel = _panel()
    original = panel.data
    result = analyze_signal_returns(
        panel,
        annualization_factors={"1D": 2, "3D": 2 / 3},
        rate_conversion_factors={"1D": 1, "3D": 1 / 3},
    )
    returns = result.get_table("factor_returns").filter(pl.col("return_column") == "1D")
    np.testing.assert_array_equal(returns["weighted_return"], [1 / 32, 1 / 32, 1 / 64])
    assert returns["n_obs"].to_list() == [2, 2, 2]
    weights = result.get_table("factor_weights")
    assert weights["weight"].to_list() == [-0.5, 0.5] * 3
    paths = result.get_table("factor_paths").filter(pl.col("return_column") == "1D")
    np.testing.assert_array_equal(
        paths["wealth"],
        [float(Fraction(33, 32)), float(Fraction(1089, 1024)), float(Fraction(70785, 65536))],
    )
    regression = (
        result.get_table("regression_statistics")
        .filter(pl.col("return_column") == "1D")
        .row(0, named=True)
    )
    assert regression["beta"] == pytest.approx(float(Fraction(7, 61)), rel=1e-13)
    assert regression["alpha_per_period"] == pytest.approx(float(Fraction(45, 1952)), rel=1e-13)
    assert regression["annual_alpha"] == pytest.approx(
        float(Fraction(1997, 1952) ** 2 - 1), rel=1e-13
    )
    assert regression["annualization_factor"] == 2
    spread = result.get_table("spread_observations").filter(pl.col("return_column") == "1D")
    np.testing.assert_array_equal(spread["spread"], [1 / 16, 1 / 16, 1 / 32])
    assert spread["standard_error"].null_count() == 3
    assert_frame_equal(panel.data, original)
    assert result.metadata["calendar"] is None


def test_return_report_exports_owned_samples_and_keeps_explicit_missing_grid_dates():
    from ml4t.diagnostic.signal import SignalReturnsResult, analyze_signal_returns

    panel = _panel()
    dates = panel.data["date"].unique().sort().to_pandas()
    grid = pd.DataFrame({"date": [dates.iloc[0] - pd.Timedelta(days=1), *dates]})
    result = analyze_signal_returns(
        panel,
        annualization_factors={"1D": 2, "3D": 2 / 3},
        date_grid=grid,
        spread_exposure="unit_gross",
    )
    dated = result.get_table("factor_returns").filter(pl.col("return_column") == "1D")
    assert dated["weighted_return"][0] is None and dated["n_obs"][0] == 0
    paths = result.get_table("factor_paths").filter(pl.col("return_column") == "1D")
    assert paths["wealth"][0] == 1
    spreads = result.get_table("spread_observations").filter(pl.col("return_column") == "1D")
    assert spreads["spread"][0] is None and spreads["standard_error"][0] is None
    np.testing.assert_array_equal(spreads["spread"].to_list()[1:], [1 / 32, 1 / 32, 1 / 64])
    restored = SignalReturnsResult.from_json(result.to_json())
    for name in result.table_names:
        assert_frame_equal(restored.get_table(name), result.get_table(name))
    owned = result.get_table("factor_returns")
    owned[1, "weighted_return"] = 999
    assert restored.get_table("factor_returns")[1, "weighted_return"] != 999


def test_return_report_converts_aggregate_mean_before_rate_and_keeps_raw_error_units():
    from ml4t.diagnostic.signal import analyze_signal_returns

    result = analyze_signal_returns(
        _panel(),
        annualization_factors={"1D": 2, "3D": 2 / 3},
        rate_conversion_factors={"1D": 1, "3D": 1 / 3},
    )
    row = (
        result.get_table("quantile_statistics")
        .filter((pl.col("return_column") == "3D") & (pl.col("quantile") == 2))
        .row(0, named=True)
    )
    independent = np.array(
        [float(Fraction(663, 8192)), float(Fraction(1279, 16384)), float(Fraction(17, 1024))]
    )
    assert row["mean"] == float(Fraction(959, 16384))
    assert row["display_mean"] == pytest.approx(
        np.cbrt(float(Fraction(17343, 16384))) - 1, abs=1e-15
    )
    assert row["display_mean"] != pytest.approx(np.mean(np.cbrt(1 + independent) - 1), abs=1e-8)
    assert row["n_obs"] == 3 and row["sample_unit"] == "date"
    assert row["standard_error"] == pytest.approx(
        np.std(independent, ddof=1) / np.sqrt(3), rel=1e-13
    )
    assert row["display_standard_error"] == pytest.approx(
        np.std(independent, ddof=1) / 3, rel=1e-13
    )
    observations = result.get_table("quantile_observations")
    assert observations["standard_error"].null_count() == observations.height
    assert observations["display_standard_error"].null_count() == observations.height


@pytest.mark.parametrize(
    "factors", [{"1D": 1}, {"1D": 0, "3D": 1}, {"1D": True, "3D": 1}, {"1D": np.inf, "3D": 1}]
)
def test_return_report_rejects_missing_or_invalid_explicit_annualization(factors):
    from ml4t.diagnostic.signal import analyze_signal_returns

    with pytest.raises(ValueError, match="annualization_factors"):
        analyze_signal_returns(_panel(), annualization_factors=factors)


def test_return_report_cannot_fit_on_dates_hidden_from_its_requested_return_grid():
    from ml4t.diagnostic.signal import analyze_signal_returns

    panel = _panel()
    clock = panel.data.select("date").unique().sort("date").slice(1)
    with pytest.raises(ValueError, match="retain every observed date"):
        analyze_signal_returns(panel, annualization_factors={"1D": 2, "3D": 2 / 3}, date_grid=clock)


@pytest.mark.parametrize("width", [375, 1200])
@pytest.mark.parametrize("profile", ["unknown", "mixed", "known_zero"])
def test_quantile_report_browser_distinguishes_unknown_zero_and_mixed_errors_without_clipping(
    width, profile, tmp_path
):
    playwright = pytest.importorskip("playwright.sync_api")
    pytest.importorskip("plotly")
    import importlib.util
    import shutil
    from pathlib import Path

    from ml4t.diagnostic import analyze_prepared_signal
    from ml4t.diagnostic.visualization import combine_figures_to_html
    from ml4t.diagnostic.visualization.signal import plot_quantile_returns_bar

    if profile == "unknown":
        result = _panel(singleton=True).analyze(min_assets=2).to_quantile_result()
        expected_bars, expected_errors = 2, 0
    else:
        dates = pd.bdate_range("2025-01-02", periods=2, tz="UTC")
        rates = [110.0, 120.0, 130.0, 140.0] if profile == "mixed" else [110.0, 110.0, 120.0, 120.0]
        quantiles = 3 if profile == "mixed" else 2
        prices = pd.DataFrame(
            {asset: [100.0, rate] for asset, rate in zip("ABCD", rates, strict=True)}, index=dates
        )
        factor = pd.Series(
            [-2.0, -1.0, 1.0, 2.0],
            index=pd.MultiIndex.from_product([dates[:1], list("ABCD")], names=["date", "asset"]),
        )
        clean = utils.get_clean_factor_and_forward_returns(
            factor, prices, periods=(1,), quantiles=quantiles, max_loss=0, filter_zscore=None
        )
        result = analyze_prepared_signal(
            clean, return_columns=["1D"], min_assets=2
        ).to_quantile_result()
        expected_bars, expected_errors = (3, 1) if profile == "mixed" else (2, 2)
    figure = plot_quantile_returns_bar(result, period="1D", show_error_bars=True)
    figure.update_layout(width=None, autosize=True)
    target = tmp_path / "quantiles.html"
    target.write_text(
        combine_figures_to_html(
            [figure],
            title="Quantile sample review",
            include_plotlyjs=True,
            sections=[
                {
                    "title": "Observed estimates",
                    "figure_index": 0,
                    "text": "<p>Unknown sample errors are unavailable. A known zero error remains defined.</p>",
                }
            ],
        ),
        encoding="utf-8",
    )
    with playwright.sync_playwright() as runtime:
        executable = None
        if importlib.util.find_spec("choreographer") is not None:
            from choreographer.browsers.chromium import Chromium

            executable = Chromium.find_browser(skip_local=False)
        executable = executable or shutil.which("google-chrome") or shutil.which("chromium")
        if executable is None and Path(runtime.chromium.executable_path).is_file():
            executable = runtime.chromium.executable_path
        assert executable is not None, "An actual Chrome/Chromium browser must be provisioned"
        browser = runtime.chromium.launch(executable_path=executable, headless=True)
        try:
            page = browser.new_page(viewport={"width": width, "height": 1000})
            page.route("http://**/*", lambda route: route.abort())
            page.route("https://**/*", lambda route: route.abort())
            page.goto(target.as_uri(), timeout=90_000)
            page.wait_for_function(
                "Boolean(document.querySelector('.js-plotly-plot')?._fullLayout)"
            )
            measured = page.evaluate("""() => {
                const rect=node=>{const r=node.getBoundingClientRect();return {left:r.left,top:r.top,right:r.right,bottom:r.bottom}};
                const plot=document.querySelector('.js-plotly-plot'),bounds=rect(plot);
                const note=[...plot.querySelectorAll('.annotation')].find(node=>node.textContent.includes('Spread Analysis'));
                const box=rect(note),bars=[...plot.querySelectorAll('.barlayer .point path')].map(rect);
                const prose=getComputedStyle(document.querySelector('.section-text p'));
                const labels=[...plot.querySelectorAll('.xtick text,.ytick text,.xtitle,.ytitle')].map(node=>({size:parseFloat(getComputedStyle(node).fontSize),family:getComputedStyle(node).fontFamily}));
                return {bars:bars.length,errorPaths:plot.querySelectorAll('.errorbar path').length,
                    metadata:plot.layout.meta,arrays:plot.data.map(trace=>trace.error_y.array),
                    proseSize:parseFloat(prose.fontSize),proseFamily:prose.fontFamily,labels,
                    overlap:bars.some(bar=>box.left<bar.right&&box.right>bar.left&&box.top<bar.bottom&&box.bottom>bar.top),
                    clipped:[...plot.querySelectorAll('.gtitle,.xtick text,.ytick text,.xtitle,.ytitle,.annotation-text')].some(node=>{const b=rect(node);return b.left<bounds.left-1||b.right>bounds.right+1||b.top<bounds.top-1||b.bottom>bounds.bottom+1}),
                    overflow:document.documentElement.scrollWidth>innerWidth};
            }""")
        finally:
            browser.close()
    assert measured["bars"] == expected_bars and measured["errorPaths"] == expected_errors
    assert measured["metadata"]["defined_error_count"] == expected_errors
    assert measured["labels"] and all(
        14 <= label["size"] <= measured["proseSize"] for label in measured["labels"]
    )
    assert all(label["family"] == measured["proseFamily"] for label in measured["labels"])
    assert not measured["overlap"] and not measured["clipped"] and not measured["overflow"], (
        measured
    )
    if profile == "unknown":
        assert measured["arrays"] == [[None, None]]
    elif profile == "known_zero":
        assert measured["arrays"] == [[0, 0]]
    else:
        assert measured["arrays"][0][1:] == [None, None]
