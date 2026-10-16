"""Factor exposure and contributions must retain the declared point-in-time universe."""

import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.metrics import compute_forward_returns
from ml4t.diagnostic.signal import prepare_signal_panel


def _inputs():
    dates = pd.bdate_range("2025-01-02", periods=5, tz="UTC")
    prices = pd.DataFrame(
        {
            "A": [128.0, 144.0, 180.0, 202.5, 253.125],
            "B": [128.0, 160.0, 140.0, 175.0, 153.125],
            "C": [128.0, 192.0, 288.0, 432.0, 648.0],
            "D": [128.0, 256.0, 288.0, 576.0, 648.0],
        },
        index=dates,
    )
    factor = (
        prices.pct_change(fill_method=None)
        .loc[dates[[1, 3]]]
        .stack(future_stack=True)
        .rename("factor")
    )
    factor.index = factor.index.set_names(["date", "asset"])
    marks = (
        prices.stack(future_stack=True).rename("close").rename_axis(["date", "asset"]).reset_index()
    )
    forward = compute_forward_returns(marks, periods=[1, 2], group_col="asset")
    forward = forward.set_index(["date", "asset"])[["fwd_ret_1", "fwd_ret_2"]].rename(
        columns={"fwd_ret_1": "1D", "fwd_ret_2": "2D"}
    )
    groups = pd.Series(
        [
            "L" if asset in (["A", "C"] if day == dates[3] else ["A", "B"]) else "H"
            for day, asset in factor.index
        ],
        index=factor.index,
        name="group",
    )
    return factor, forward, groups, dates


def _expected(demeaned, group_adjust, equal_weight, second, dated):
    if group_adjust:
        if demeaned:
            return [-0.25, -0.25, 0.25, 0.25] if second and dated else [-0.25, 0.25, -0.25, 0.25]
        if equal_weight:
            return [0.25] * 4
        return [0.1, 0.1, 0.4, 0.4] if second and dated else [1 / 6, 1 / 3, 1 / 6, 1 / 3]
    if equal_weight:
        return [-0.25, -0.25, 0.25, 0.25] if demeaned else [0.25] * 4
    return [-11 / 36, -7 / 36, 1 / 36, 17 / 36] if demeaned else [1 / 15, 2 / 15, 4 / 15, 8 / 15]


@pytest.mark.parametrize("representation", ["multiindex", "pandas", "polars"])
@pytest.mark.parametrize("dated", [False, True])
@pytest.mark.parametrize("demeaned", [False, True])
@pytest.mark.parametrize("group_adjust", [False, True])
@pytest.mark.parametrize("equal_weight", [False, True])
def test_factor_weights_match_independent_exposure_not_return_demeaning_or_bucket_labels(
    representation, dated, demeaned, group_adjust, equal_weight
):
    factor, forward, groups, dates = _inputs()
    if not dated:
        groups = groups.groupby(level="asset").first().to_dict()
    forward = forward[["1D"]]
    if representation != "multiindex":
        factor, forward = factor.reset_index(), forward.reset_index()
        if dated:
            groups = groups.reset_index()
        if representation == "polars":
            factor, forward = pl.from_pandas(factor), pl.from_pandas(forward)
            if dated:
                groups = pl.from_pandas(groups)
    panel = prepare_signal_panel(factor, forward, groups=groups, quantiles=2, max_loss=0)
    before = panel.data
    weights = panel.factor_weights(
        demeaned=demeaned, group_adjust=group_adjust, equal_weight=equal_weight
    )
    assert weights.select("date", "asset").equals(before.select("date", "asset"))
    for second, day in enumerate(dates[[1, 3]]):
        one = weights.filter(pl.col("date") == day).sort("asset")
        expected = _expected(demeaned, group_adjust, equal_weight, second, dated)
        assert one["weight"].to_list() == pytest.approx(expected, abs=1e-14)
        assert one["weight"].abs().sum() == pytest.approx(1)
        if demeaned:
            assert one["weight"].sum() == pytest.approx(0, abs=1e-14)
        if group_adjust:
            assert one.group_by("group").agg(pl.col("weight").abs().sum())[
                "weight"
            ].to_list() == pytest.approx([0.5, 0.5])
    assert panel.data.equals(before)


@pytest.mark.parametrize("demeaned", [False, True])
@pytest.mark.parametrize("group_adjust", [False, True])
@pytest.mark.parametrize("equal_weight", [False, True])
@pytest.mark.parametrize("by_asset", [False, True])
def test_factor_return_contributions_use_the_same_weights_and_actual_supplied_outcomes(
    demeaned, group_adjust, equal_weight, by_asset
):
    factor, forward, groups, dates = _inputs()
    panel = prepare_signal_panel(factor, forward[["1D"]], groups=groups, quantiles=2, max_loss=0)
    result = panel.factor_returns(
        demeaned=demeaned, group_adjust=group_adjust, equal_weight=equal_weight, by_asset=by_asset
    )
    outcomes = [0.25, -0.125, 0.5, 0.125]
    for second, day in enumerate(dates[[1, 3]]):
        actual = result.filter(pl.col("date") == day)
        expected_weights = _expected(demeaned, group_adjust, equal_weight, second, True)
        expected = [
            weight * value for weight, value in zip(expected_weights, outcomes, strict=True)
        ]
        if by_asset:
            assert actual.sort("asset")["weighted_return"].to_list() == pytest.approx(
                expected, abs=1e-14
            )
            assert actual["n_obs"].to_list() == [1] * 4
        else:
            assert actual["weighted_return"].to_list() == pytest.approx([sum(expected)], abs=1e-14)
            assert actual["n_obs"].to_list() == [4]
    assert result["return_column"].unique().to_list() == ["1D"]
