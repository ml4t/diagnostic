"""Summary/full adapters must preserve actual reference tables and composition flags."""

import os
import runpy
from fractions import Fraction
from importlib.metadata import version
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _panel():
    return runpy.run_path(str(Path(__file__).with_name("test_alphalens_information_sheet.py")))[
        "_price_panel"
    ]()


def test_required_reference_job_cannot_pass_without_an_executable_return_oracle():
    required = os.environ.get("ML4T_REQUIRED_ALPHALENS_REFERENCE")
    if required is None:
        pytest.skip("Explicit required-reference job, not an ordinary optional-reference run")
    import alphalens.performance as performance
    import alphalens.plotting
    import alphalens.tears

    assert version("alphalens-reloaded") == required
    data = _panel()
    actual = performance.factor_returns(data)
    expected = sum(
        Fraction(weight, 18) * Fraction(asset + 1, 100 + 10 * asset)
        for asset, weight in enumerate([-5, -3, -1, 1, 3, 5])
    )
    assert actual.index[0] == data.index.get_level_values("date")[0]
    assert actual.iloc[0]["1D"] == pytest.approx(float(expected), rel=1e-13)


@pytest.mark.parametrize(
    "sheet,by_group,count", [("summary", False, 2), ("full", False, 20), ("full", True, 23)]
)
def test_familiar_composition_keeps_every_component_and_reference_information_adjustment(
    sheet, by_group, count, tmp_path
):
    pytest.importorskip("plotly")
    from ml4t.diagnostic.compat.alphalens import tears

    data = _panel()
    before = data.copy(deep=True)
    route = getattr(tears, f"create_{sheet}_tear_sheet")
    options = {"by_group": by_group} if sheet == "full" else {}
    target = tmp_path / "composed.html"
    figures = route(
        data, group_neutral=True, return_fig=True, output_file=target, set_context=False, **options
    )
    assert len(figures) == count
    html = target.read_text(encoding="utf-8")
    assert html.count('class="plot-container"') == count
    assert f"information group adjustment: {sheet == 'full'}" in html
    for identifier in (
        "factor-statistics",
        "return-regression",
        "quantile-means",
        "observed-spreads",
        "information-summary",
        "turnover-summary",
        "rank-summary",
    ):
        assert len(pd.read_html(StringIO(html), attrs={"id": identifier})) == 1
    pd.testing.assert_frame_equal(data, before)
    target.write_text("retained report", encoding="utf-8")
    with pytest.raises(TypeError, match="group_neutral"):
        route(data, group_neutral="yes", output_file=target)
    assert target.read_text(encoding="utf-8") == "retained report"


@pytest.mark.parametrize("sheet,by_group", [("summary", False), ("full", False), ("full", True)])
@pytest.mark.parametrize("long_short", [False, True])
@pytest.mark.parametrize("group_neutral", [False, True])
def test_complete_reference_summary_full_workflows_retain_factor_return_information_and_rank_tables(
    sheet, by_group, long_short, group_neutral, monkeypatch, tmp_path
):
    reference = pytest.importorskip("alphalens.tears")
    plotting = pytest.importorskip("alphalens.plotting")
    pyplot = pytest.importorskip("matplotlib.pyplot")
    profile = version("alphalens-reloaded")
    if profile not in {"0.5.0", "0.4.6"}:
        pytest.skip("Corrected and frozen reference profiles are qualified independently")
    from ml4t.diagnostic.compat.alphalens import tears

    data = _panel()
    before = data.copy(deep=True)
    captured, displayed = {}, []
    original = {
        name: getattr(plotting, name)
        for name in [
            "plot_quantile_statistics_table",
            "plot_returns_table",
            "plot_information_table",
            "plot_turnover_table",
        ]
    }

    def factor_table(sample, *args, **kwargs):
        captured["factor"] = original["plot_quantile_statistics_table"](sample, return_df=True)
        return original["plot_quantile_statistics_table"](sample, *args, **kwargs)

    def return_table(alpha_beta, means, spread, *args, **kwargs):
        captured["regression"] = alpha_beta.copy(deep=True)
        captured["means"] = means.copy(deep=True)
        captured["spread"] = spread.copy(deep=True)
        return original["plot_returns_table"](alpha_beta, means, spread, *args, **kwargs)

    def information_table(ic, *args, **kwargs):
        captured["information"] = original["plot_information_table"](ic, return_df=True)
        return original["plot_information_table"](ic, *args, **kwargs)

    def turnover_table(rank, membership, *args, **kwargs):
        captured["turnover"], captured["rank"] = original["plot_turnover_table"](
            rank, membership, return_df=True
        )
        return original["plot_turnover_table"](rank, membership, *args, **kwargs)

    def show():
        for identifier in pyplot.get_fignums():
            figure = pyplot.figure(identifier)
            figure.canvas.draw()
            displayed.append(
                sum(
                    len(axis.lines) + len(axis.patches) + len(axis.collections)
                    for axis in figure.axes
                )
            )

    for name, wrapper in [
        ("plot_quantile_statistics_table", factor_table),
        ("plot_returns_table", return_table),
        ("plot_information_table", information_table),
        ("plot_turnover_table", turnover_table),
    ]:
        monkeypatch.setattr(plotting, name, wrapper)
    monkeypatch.setattr(pyplot, "show", show)
    options = {"by_group": by_group} if sheet == "full" else {}
    try:
        getattr(reference, f"create_{sheet}_tear_sheet")(
            data, long_short=long_short, group_neutral=group_neutral, set_context=False, **options
        )
    finally:
        pyplot.close("all")
    assert displayed and all(count > 0 for count in displayed)
    assert set(captured) == {
        "factor",
        "regression",
        "means",
        "spread",
        "information",
        "turnover",
        "rank",
    }
    target = tmp_path / "candidate.html"
    getattr(tears, f"create_{sheet}_tear_sheet")(
        data,
        long_short=long_short,
        group_neutral=group_neutral,
        reference_version=profile,
        output_file=target,
        return_fig=True,
        set_context=False,
        **options,
    )
    html = target.read_text(encoding="utf-8")

    def table(identifier):
        return pd.read_html(StringIO(html), attrs={"id": identifier})[0]

    def same_display(actual, expected):
        np.testing.assert_allclose(
            actual,
            [float(f"{value:.6g}") for value in np.asarray(expected).reshape(-1)],
            rtol=1e-14,
            atol=1e-14,
            equal_nan=True,
        )

    factor = table("factor-statistics").set_index("quantile")
    for field in ("min", "max", "mean", "std", "count"):
        same_display(factor[field], captured["factor"][field])
    same_display(factor["count_percent"], captured["factor"]["count %"])
    regression = table("return-regression").set_index("return_column")
    means = table("quantile-means")
    spread = table("observed-spreads").set_index("return_column")
    information = table("information-summary").set_index("return_column")
    for column in ("1D", "2D"):
        same_display(
            [regression.loc[column, "annual_alpha"], regression.loc[column, "beta"]],
            captured["regression"][column],
        )
        same_display(
            means.loc[means["return_column"] == column, "display_mean"], captured["means"][column]
        )
        same_display([spread.loc[column, "mean_spread"]], [captured["spread"][column].mean()])
        for actual, expected in [
            ("mean_ic", "IC Mean"),
            ("std_ic", "IC Std."),
            ("risk_adjusted_ic", "Risk-Adjusted IC"),
            ("t_stat", "t-stat(IC)"),
            ("p_value", "p-value(IC)"),
            ("skew", "IC Skew"),
            ("kurtosis", "IC Kurtosis"),
        ]:
            same_display(
                [information.loc[column, actual]], [captured["information"].loc[column, expected]]
            )
    membership = table("turnover-summary").set_index(["lag", "quantile"])
    rank = table("rank-summary").set_index("lag")
    for lag in (1, 2):
        for bucket in (1, 2, 3):
            same_display(
                [membership.loc[(lag, bucket), "turnover"]],
                [captured["turnover"].loc[f"Quantile {bucket} Mean Turnover ", f"{lag}D"]],
            )
        same_display(
            [rank.loc[lag, "autocorrelation"]],
            [captured["rank"].loc["Mean Factor Rank Autocorrelation", f"{lag}D"]],
        )
    pd.testing.assert_frame_equal(data, before)
