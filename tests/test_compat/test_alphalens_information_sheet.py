"""The familiar information sheet must retain the prepared price-produced sample."""

from fractions import Fraction

import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy.stats import rankdata


def _price_panel():
    from ml4t.diagnostic.compat.alphalens import utils

    dates = pd.date_range("2025-01-02T00:00:00.000000001", periods=30, tz="UTC", unit="ns")
    assets = list("ABCDEF")
    prices = pd.DataFrame(
        {asset: [100 + 10 * i + (i + 1) * j for j in range(30)] for i, asset in enumerate(assets)},
        index=dates,
    )
    factor = pd.Series(
        np.concatenate([np.roll(np.arange(1.0, 7.0), j % 6) for j in range(28)]),
        index=pd.MultiIndex.from_product([dates[:28], assets], names=["date", "asset"]),
        name="factor",
    )
    groups = {asset: "first" if i < 2 else "second" for i, asset in enumerate(assets)}
    return utils.get_clean_factor_and_forward_returns(
        factor,
        prices,
        periods=(1, 2),
        groupby=groups,
        quantiles=3,
        filter_zscore=None,
        max_loss=0,
    )


@pytest.mark.parametrize("group_neutral", [False, True])
@pytest.mark.parametrize("by_group", [False, True])
def test_familiar_price_to_information_sheet_preserves_global_ic_and_all_group_options(
    group_neutral, by_group, tmp_path
):
    from ml4t.diagnostic.compat.alphalens import tears

    panel = _price_panel()
    before = panel.copy(deep=True)
    target = tmp_path / "information.html"
    figures = tears.create_information_tear_sheet(
        panel,
        group_neutral=group_neutral,
        by_group=by_group,
        output_file=target,
        return_fig=True,
        set_context=False,
    )
    assert len(figures) == (7 if by_group else 8)
    expected = []
    for _, sample in panel.groupby(level="date", observed=True):
        outcomes = sample["1D"]
        if group_neutral:
            outcomes = outcomes - outcomes.groupby(sample["group"], observed=True).transform("mean")
        x, y = rankdata(sample["factor"]), rankdata(outcomes)
        x, y = x - x.mean(), y - y.mean()
        expected.append(float(np.sum(x * y) / np.sqrt(np.sum(x * x) * np.sum(y * y))))
    np.testing.assert_allclose(figures[0].data[0].y, expected, atol=1e-14, rtol=0)
    pd.testing.assert_frame_equal(panel, before)
    assert panel.iloc[0]["1D"] == pytest.approx(float(Fraction(1, 100)), rel=3e-16)
    html = target.read_text(encoding="utf-8")
    assert html.count('class="plot-container"') == (7 if by_group else 8)
    assert "Defined ICs" in html and "dimensionless" in html
    assert "Mean IC by group" in html if by_group else "Monthly" in html


def test_owned_prepared_constructor_preserves_sparse_buckets_groups_keys_and_calendar_null():
    from ml4t.diagnostic.signal import PreparedSignalPanel

    original = _price_panel()
    original["factor_quantile"] = original["factor_quantile"].map({1: 2, 2: 5, 3: 9})
    panel = PreparedSignalPanel.from_prepared(original, return_columns=["1D", "2D"], calendar=None)
    assert panel.metadata["bucket_source"] == "supplied, not recomputed"
    assert panel.metadata["bucket_ids"] == [2, 5, 9]
    assert panel.metadata["calendar"] is None
    assert panel.metadata["calendar_source"] == "argument"
    assert panel.excluded_data.is_empty() and panel.outcome_data is None
    assert panel.data.height == original.shape[0]
    assert set(panel.data["group"]) == {"first", "second"}
    owned = panel.data
    original.iloc[0, original.columns.get_loc("factor")] = 999
    owned[0, "factor"] = -999
    assert panel.data["factor"][0] == 1
    assert panel.ic_by_date()["n_obs"].to_list() == [6] * 56
    assert panel.data["date"].dtype == pl.Datetime("ns", "UTC")


def test_information_adapter_does_not_replace_a_report_for_invalid_prepared_input(tmp_path):
    from ml4t.diagnostic.compat.alphalens import tears

    panel = _price_panel()
    panel.iloc[0, panel.columns.get_loc("1D")] = np.nan
    target = tmp_path / "information.html"
    target.write_text("retained report", encoding="utf-8")
    with pytest.raises(ValueError, match="complete and finite"):
        tears.create_information_tear_sheet(panel, output_file=target)
    assert target.read_text(encoding="utf-8") == "retained report"
