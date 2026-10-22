"""A prepared-price return report must retain samples, scale and missing uncertainty."""

from fractions import Fraction
from importlib.metadata import version

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
    membership_counts = clean.groupby("factor_quantile", observed=True).size()
    assert sorted(membership_counts) == [1, 1, 2]
    known = [f"Q{bucket}" for bucket in membership_counts[membership_counts == 2].index]
    assert list(figure.data[0].x) == ["Q1", "Q2", "Q3"]
    assert len(figure.data) == 2 and not figure.data[0].error_y.visible
    assert list(figure.data[1].x) == known
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


def test_prepared_sparse_bucket_spread_selects_actual_extremes_not_bucket_count():
    pytest.importorskip("plotly")
    from ml4t.diagnostic import analyze_prepared_signal
    from ml4t.diagnostic.visualization.signal import plot_spread_timeseries

    data = _panel().data.with_columns(pl.col("quantile").replace_strict({1: 2, 2: 9}))
    model = analyze_prepared_signal(
        data, return_columns=["1D", "3D"], min_assets=2
    ).to_quantile_result()
    figure = plot_spread_timeseries(model, period="1D")
    np.testing.assert_allclose(figure.data[0].y, [1 / 16, 1 / 16, 1 / 32])
    with pytest.raises(ValueError, match="quantile|bucket"):
        plot_spread_timeseries(model, period="1D", lower_quantile=1)


def test_owned_return_chart_bridge_retains_date_counts_display_errors_and_missing_inference():
    from ml4t.diagnostic.signal import analyze_signal_returns

    result = analyze_signal_returns(
        _panel(),
        annualization_factors={"1D": 2, "3D": 2 / 3},
        rate_conversion_factors={"1D": 1, "3D": 1 / 3},
    )
    model = result.to_quantile_result("3D")
    assert model.count_by_quantile == {"Q1": 3, "Q2": 3}
    assert model.mean_returns["3D"]["Q2"] == pytest.approx((17343 / 16384) ** (1 / 3) - 1)
    raw = np.array([663 / 8192, 1279 / 16384, 17 / 1024])
    assert model.std_returns["3D"]["Q2"] == pytest.approx(np.std(raw, ddof=1) / np.sqrt(3))
    assert model.get_monotonicity("3D") == (None, "unavailable")
    assert np.isnan(model.spread_t_stat["3D"])
    assert model.get_dataframe("observations")["n_obs"].to_list() == [1] * 6
    with pytest.raises(ValueError, match="return|period|column"):
        result.to_quantile_result("absent")


def test_owned_returns_renderer_exposes_every_selected_horizon_and_never_compounds_display_rates(
    tmp_path,
):
    pytest.importorskip("plotly")
    from ml4t.diagnostic.signal import analyze_signal_returns
    from ml4t.diagnostic.visualization.signal import (
        generate_signal_returns_report,
        plot_signal_returns,
    )

    result = analyze_signal_returns(
        _panel(),
        annualization_factors={"1D": 2, "3D": 2 / 3},
        rate_conversion_factors={"1D": 1, "3D": 1 / 3},
        spread_exposure="unit_gross",
    )
    figures = plot_signal_returns(result, rolling_window=2, bandwidth=0.5)
    assert len(figures) == 10
    for column in ("1D", "3D"):
        selected = [figure for figure in figures if figure.layout.meta["return_column"] == column]
        assert len(selected) == 5
        paths = result.get_table("quantile_paths").filter(
            (pl.col("return_column") == column) & (pl.col("quantile") == 2)
        )
        cumulative = next(
            figure
            for figure in selected
            if figure.layout.meta["analysis"] == "quantile_cumulative_returns"
        )
        np.testing.assert_allclose(cumulative.data[1].y, paths["wealth"] - 1)
        spread = next(
            figure for figure in selected if figure.layout.meta["analysis"] == "quantile_spread"
        )
        expected = result.get_table("spread_observations").filter(pl.col("return_column") == column)
        np.testing.assert_allclose(spread.data[0].y, expected["spread"])
        assert spread.layout.meta["exposure_scale"] == 0.5
        assert spread.layout.meta["bandwidth"] == 0.5
    target = tmp_path / "returns.html"
    assert generate_signal_returns_report(result, output_file=target) == str(target)
    html = target.read_text(encoding="utf-8")
    assert html.count('class="plot-container"') == 10
    assert "annual_alpha" in html and "beta" in html
    assert "dimensionless" in html and "not self-financing" in html
    assert "sample standard error" in html
    target.write_text("retained", encoding="utf-8")
    with pytest.raises(ValueError, match="rolling_window"):
        generate_signal_returns_report(result, output_file=target, rolling_window=0)
    assert target.read_text(encoding="utf-8") == "retained"


def test_owned_return_cumulative_charts_retain_nanosecond_keys_and_group_means():
    pytest.importorskip("plotly")
    import runpy
    from pathlib import Path

    from ml4t.diagnostic.signal import analyze_signal_returns
    from ml4t.diagnostic.visualization.signal import plot_signal_returns

    fixture = (
        Path(__file__).resolve().parents[1] / "test_compat/test_alphalens_information_sheet.py"
    )
    data = runpy.run_path(str(fixture))["_price_panel"]()
    panel = PreparedSignalPanel.from_prepared(data, return_columns=["1D", "2D"], calendar=None)
    result = analyze_signal_returns(
        panel, annualization_factors={"1D": 252, "2D": 126}, by_group=True
    )
    figures = plot_signal_returns(result)
    assert len(figures) == 14
    cumulative = next(
        figure
        for figure in figures
        if figure.layout.meta["analysis"] == "quantile_cumulative_returns"
    )
    expected = data.index.get_level_values("date").unique()
    assert [pd.Timestamp(value).value for value in cumulative.data[0].x] == expected.asi8.tolist()
    assert cumulative.layout.xaxis.type == "category"
    groups = [
        figure for figure in figures if figure.layout.meta["analysis"] == "group_quantile_mean"
    ]
    assert len(groups) == 4
    for figure in groups:
        sample = (
            result.get_table("group_quantile_statistics")
            .filter(
                (pl.col("group") == figure.layout.meta["group"])
                & (pl.col("return_column") == figure.layout.meta["return_column"])
            )
            .sort("quantile")
        )
        np.testing.assert_allclose(figure.data[0].y, sample["display_mean"])


@pytest.mark.parametrize("width", [375, 1200])
def test_complete_owned_returns_browser_keeps_all_panels_readable_offline(width, tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    pytest.importorskip("plotly")
    import importlib.util
    import shutil
    from pathlib import Path

    from ml4t.diagnostic.signal import analyze_signal_returns
    from ml4t.diagnostic.visualization.signal import generate_signal_returns_report

    result = analyze_signal_returns(_panel(), annualization_factors={"1D": 2, "3D": 2 / 3})
    target = tmp_path / "returns.html"
    generate_signal_returns_report(result, output_file=target)
    with playwright.sync_playwright() as runtime:
        executable = None
        if importlib.util.find_spec("choreographer"):
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
                "document.querySelectorAll('.js-plotly-plot').length===10 && [...document.querySelectorAll('.js-plotly-plot')].every(p=>p._fullLayout)"
            )
            measured = page.evaluate("""() => {
                const prose=getComputedStyle(document.querySelector('.section-text'));
                return {proseSize:parseFloat(prose.fontSize),proseFamily:prose.fontFamily,
                    overflow:document.documentElement.scrollWidth>innerWidth,
                    plots:[...document.querySelectorAll('.js-plotly-plot')].map(plot=>{
                        const box=plot.getBoundingClientRect();
                        const labels=[...plot.querySelectorAll('.xtick text,.ytick text,.xtitle,.ytitle,.legendtext')].filter(n=>n.textContent.trim()).map(n=>({size:parseFloat(getComputedStyle(n).fontSize),family:getComputedStyle(n).fontFamily}));
                        const clipped=[...plot.querySelectorAll('.gtitle,.xtick text,.ytick text,.xtitle,.ytitle,.annotation-text,.legendtext')].filter(n=>{const b=n.getBoundingClientRect();return b.width>0&&b.height>0&&(b.left<box.left-1||b.right>box.right+1||b.top<box.top-1||b.bottom>box.bottom+1)}).map(n=>n.textContent);
                        return {analysis:plot.layout.meta.analysis,return_column:plot.layout.meta.return_column,labels,clipped};
                    })};
            }""")
        finally:
            browser.close()
    assert not measured["overflow"] and len(measured["plots"]) == 10
    assert {plot["return_column"] for plot in measured["plots"]} == {"1D", "3D"}
    for plot in measured["plots"]:
        assert plot["labels"] and not plot["clipped"], plot
        assert all(
            14 <= label["size"] <= measured["proseSize"]
            and label["family"] == measured["proseFamily"]
            for label in plot["labels"]
        ), plot


@pytest.mark.parametrize("demeaned", [False, True])
@pytest.mark.parametrize("group_adjust", [False, True])
@pytest.mark.parametrize("by_group", [False, True])
def test_return_report_matches_reference_weights_regression_and_exact_quantile_samples(
    demeaned, group_adjust, by_group
):
    reference = pytest.importorskip("alphalens.performance")
    if version("alphalens-reloaded") not in {"0.5.0", "0.4.6"}:
        pytest.skip("Corrected and frozen reference profiles are qualified separately")
    import runpy
    from pathlib import Path

    from ml4t.diagnostic.signal import analyze_signal_returns

    fixture = (
        Path(__file__).resolve().parents[1] / "test_compat/test_alphalens_information_sheet.py"
    )
    data = runpy.run_path(str(fixture))["_price_panel"]()
    columns = ["1D", "2D"]
    panel = PreparedSignalPanel.from_prepared(data, return_columns=columns, calendar=None)
    result = analyze_signal_returns(
        panel,
        annualization_factors={"1D": 252, "2D": 126},
        demeaned=demeaned,
        group_adjust=group_adjust,
        by_group=by_group,
        rate_conversion_factors={"1D": 1, "2D": 0.5},
    )
    weights = reference.factor_weights(data, demeaned=demeaned, group_adjust=group_adjust)
    actual = result.get_table("factor_weights").to_pandas().set_index(["date", "asset"])["weight"]
    pd.testing.assert_index_equal(actual.index, weights.index)
    np.testing.assert_allclose(actual, weights, rtol=1e-12, atol=1e-12)
    returns = reference.factor_returns(data, demeaned=demeaned, group_adjust=group_adjust)
    actual = (
        result.get_table("factor_returns")
        .to_pandas()
        .pivot(index="date", columns="return_column", values="weighted_return")[columns]
    )
    pd.testing.assert_index_equal(actual.index, returns.index)
    np.testing.assert_allclose(actual, returns, rtol=1e-12, atol=1e-12)
    regression = reference.factor_alpha_beta(
        data, returns=returns, demeaned=demeaned, group_adjust=group_adjust
    )
    for row in result.get_table("regression_statistics").iter_rows(named=True):
        np.testing.assert_allclose(
            [row["annual_alpha"], row["beta"]],
            regression[row["return_column"]].to_numpy(),
            rtol=1e-12,
            atol=1e-12,
        )
    for table, grouped, dated in [
        ("quantile_statistics", False, False),
        ("quantile_observations", False, True),
        *([("group_quantile_statistics", True, False)] if by_group else []),
    ]:
        means, errors = reference.mean_return_by_quantile(
            data, by_date=dated, by_group=grouped, demeaned=demeaned, group_adjust=group_adjust
        )
        # Pandas retains nanoseconds; Python datetimes from Polars rows do not.
        for row in result.get_table(table).to_pandas().to_dict(orient="records"):
            key = (
                (row["quantile"], row["date"])
                if dated
                else (row["quantile"], row["group"])
                if grouped
                else row["quantile"]
            )
            column = row["return_column"]
            np.testing.assert_allclose(row["mean"], means.loc[key, column], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(
                np.nan if row["standard_error"] is None else row["standard_error"],
                errors.loc[key, column],
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            )
    raw_daily, _ = reference.mean_return_by_quantile(
        data, by_date=True, demeaned=demeaned, group_adjust=group_adjust
    )
    for column in columns:
        path = result.get_table("factor_paths").filter(pl.col("return_column") == column)
        np.testing.assert_allclose(
            path["wealth"], reference.cumulative_returns(returns[column]), rtol=1e-12, atol=1e-12
        )
        for quantile in (1, 2, 3):
            actual = result.get_table("quantile_paths").filter(
                (pl.col("return_column") == column) & (pl.col("quantile") == quantile)
            )
            expected = reference.cumulative_returns(
                raw_daily.xs(quantile, level="factor_quantile")[column]
            )
            np.testing.assert_allclose(actual["wealth"], expected, rtol=1e-12, atol=1e-12)


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
        mixed_counts = None
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
        mixed_counts = clean.groupby("factor_quantile", observed=True).size()
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
        assert sorted(mixed_counts) == [1, 1, 2]
        for index, bucket in enumerate([1, 2, 3]):
            if mixed_counts[bucket] == 2:
                assert measured["arrays"][0][index] == pytest.approx(0.05)
            else:
                assert measured["arrays"][0][index] is None
