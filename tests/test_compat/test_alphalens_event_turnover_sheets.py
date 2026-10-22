"""Familiar report calls must preserve real prepared samples and supplied level units."""

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.compat.alphalens import tears, utils


def _inputs():
    dates = pd.bdate_range("2025-01-02", periods=14, tz="UTC")
    assets = list("ABCDEF")
    changes = np.array([np.roll([-0.04, -0.02, 0.01, 0.02, 0.04, 0.06], day) for day in range(13)])
    prices = pd.DataFrame(
        np.vstack([np.full(6, 100.0), 100 * np.cumprod(1 + changes, axis=0)]),
        index=dates,
        columns=assets,
    )
    factor = prices.pct_change(fill_method=None).iloc[1:9].stack(future_stack=True)
    factor.index = factor.index.set_names(["date", "asset"])
    panel = utils.get_clean_factor_and_forward_returns(
        factor,
        prices,
        periods=(1, 2),
        quantiles=2,
        groupby=dict(zip(assets, ["left"] * 3 + ["right"] * 3, strict=True)),
        filter_zscore=None,
        max_loss=0,
    )
    return panel, prices


@pytest.mark.parametrize("route", ["turnover", "event_returns", "event_study"])
def test_price_preparation_to_familiar_reports_retains_every_panel_and_input(route, tmp_path):
    pytest.importorskip("plotly")
    panel, prices = _inputs()
    before, original_prices = panel.copy(deep=True), prices.copy(deep=True)
    args = (panel,) if route == "turnover" else (panel, prices)
    options = {"turnover_periods": ["1D", "2D"]} if route == "turnover" else {"avgretplot": (2, 3)}
    if route == "event_returns":
        options["by_group"] = True
    if route == "event_study":
        options["n_bars"] = 5
    target = tmp_path / (route + ".html")
    figures = getattr(tears, f"create_{route}_tear_sheet")(
        *args, **options, output_file=target, return_fig=True, set_context=False
    )
    expected = {"turnover": 4, "event_returns": 5, "event_study": 6}[route]
    assert len(figures) == expected and all(len(figure.data) for figure in figures)
    assert target.read_text().count('class="plot-container"') == expected
    pd.testing.assert_frame_equal(panel, before)
    pd.testing.assert_frame_equal(prices, original_prices)
    assert len(panel) == 48 and list(utils.get_forward_returns_columns(panel.columns)) == [
        "1D",
        "2D",
    ]


def test_event_return_adapter_preserves_supplied_levels_instead_of_compounding_them():
    pytest.importorskip("plotly")
    dates = pd.bdate_range("2025-01-02", periods=3, tz="UTC")
    prices = pd.DataFrame({"A": [10.0, 11.0, 12.0], "B": [20.0, 22.0, 24.0]}, index=dates)
    factor = pd.Series(
        1.0,
        index=pd.MultiIndex.from_product([dates[:2], ["A", "B"]], names=["date", "asset"]),
    )
    panel = utils.get_clean_factor_and_forward_returns(
        factor, prices, periods=(1,), quantiles=1, filter_zscore=None, max_loss=0
    )
    figures = tears.create_event_returns_tear_sheet(
        panel, prices, avgretplot=(0, 1), long_short=False, std_bar=False, return_fig=True
    )
    assert len(figures) == 1
    np.testing.assert_allclose(figures[0].data[0].y, [15.75, 17.25], rtol=0, atol=1e-14)
    assert figures[0].layout.meta["units"] == "supplied_levels"


@pytest.mark.parametrize("route", ["turnover", "event_returns", "event_study"])
def test_invalid_prepared_sample_cannot_overwrite_existing_familiar_report(route, tmp_path):
    pytest.importorskip("plotly")
    panel, prices = _inputs()
    panel.iloc[0, panel.columns.get_loc("1D")] = np.nan
    target = tmp_path / "retained.html"
    target.write_text("Retain the previous report", encoding="utf-8")
    args = (panel,) if route == "turnover" else (panel, prices)
    with pytest.raises(ValueError, match="complete and finite"):
        getattr(tears, f"create_{route}_tear_sheet")(*args, output_file=target)
    assert target.read_text(encoding="utf-8") == "Retain the previous report"
