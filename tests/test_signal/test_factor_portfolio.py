"""Factor exposure and contributions must retain the declared point-in-time universe."""

import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.metrics import compute_forward_returns
from ml4t.diagnostic.signal import prepare_signal_panel


def _inputs(nanoseconds=False, zero_quote=False):
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
    if zero_quote:
        prices.loc[dates[2], "A"] = 0.0
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


@pytest.mark.parametrize("profile", ["constant", "zero", "singleton"])
@pytest.mark.parametrize("demeaned", [False, True])
@pytest.mark.parametrize("group_adjust", [False, True])
@pytest.mark.parametrize("equal_weight", [False, True])
def test_no_signal_has_explicit_zero_exposure_not_undefined_released_weights(
    profile, demeaned, group_adjust, equal_weight
):
    factor, forward, groups, dates = _inputs()
    if profile == "constant":
        factor = factor.groupby(level="date").transform(lambda values: values.mean())
    elif profile == "zero":
        factor = factor.groupby(level="date").transform(lambda values: values - values)
    else:
        factor = factor[factor.index.get_level_values("asset") == "A"]
    panel = prepare_signal_panel(factor, forward[["1D"]], groups=groups, quantiles=1, max_loss=0)
    weights = panel.factor_weights(
        demeaned=demeaned, group_adjust=group_adjust, equal_weight=equal_weight
    )
    active = not demeaned and profile != "zero"
    assets = 1 if profile == "singleton" else 4
    assert weights["weight"].to_list() == pytest.approx(
        [1 / assets if active else 0] * (2 * assets)
    )
    assert weights["zero_weight_policy"].unique().to_list() == ["zero"]
    returns = panel.factor_returns(
        demeaned=demeaned, group_adjust=group_adjust, equal_weight=equal_weight
    )
    assert returns["n_obs"].to_list() == [assets, assets]
    if not active:
        assert returns["weighted_return"].to_list() == [0, 0]
    assert weights["date"].dt.timestamp("ns").unique().sort().to_list() == [
        dates[1].value,
        dates[3].value,
    ]


@pytest.mark.parametrize("equal_weight", [False, True])
def test_inactive_group_receives_no_weight_while_active_group_retains_unit_gross(equal_weight):
    factor, forward, groups, dates = _inputs()
    factor.loc[(dates[1], ["A", "B"])] = 0.25
    panel = prepare_signal_panel(factor, forward[["1D"]], groups=groups, quantiles=1, max_loss=0)
    result = panel.factor_weights(group_adjust=True, equal_weight=equal_weight)
    first = result.filter(pl.col("date") == dates[1]).sort("asset")
    assert first["weight"].to_list() == pytest.approx([0, 0, -0.5, 0.5])


@pytest.mark.parametrize("equal_weight", [False, True])
def test_unequal_groups_and_singleton_do_not_allocate_to_zero_dispersion(equal_weight):
    factor, forward, _, dates = _inputs()
    groups = {"A": "large", "B": "large", "C": "large", "D": "singleton"}
    panel = prepare_signal_panel(factor, forward[["1D"]], groups=groups, quantiles=2, max_loss=0)
    weights = panel.factor_weights(group_adjust=True, equal_weight=equal_weight)
    for day in dates[[1, 3]]:
        assert weights.filter(pl.col("date") == day).sort("asset")[
            "weight"
        ].to_list() == pytest.approx(
            [-0.5, 0, 0.5, 0] if equal_weight else [-0.4, -0.1, 0.5, 0], abs=1e-14
        )


@pytest.mark.parametrize("by_asset", [False, True])
def test_actual_missing_future_tail_does_not_reallocate_weights_or_invent_zero_returns(by_asset):
    from ml4t.diagnostic.signal import compute_factor_returns

    factor, forward, groups, dates = _inputs()
    raw = factor.to_frame().join(forward).join(groups)
    before = raw.copy(deep=True)
    result = compute_factor_returns(raw, return_columns=["2D", "1D"], by_asset=by_asset)
    assert result["return_column"].unique(maintain_order=True).to_list() == ["2D", "1D"]
    tail = result.filter((pl.col("date") == dates[3]) & (pl.col("return_column") == "2D"))
    assert tail["weighted_return"].null_count() == len(tail)
    assert tail["n_obs"].to_list() == [0] * len(tail)
    if by_asset:
        assert tail.sort("asset")["weight"].to_list() == pytest.approx(
            [-11 / 36, -7 / 36, 1 / 36, 17 / 36]
        )
    else:
        first = result.filter(pl.col("return_column") == "1D")
        assert first["weighted_return"].to_list() == pytest.approx([1 / 48, 1 / 48])
    pd.testing.assert_frame_equal(raw, before)


def test_factor_exposure_and_contributions_preserve_quote_nanoseconds_and_snapshot_ownership():
    factor, forward, groups, dates = _inputs(nanoseconds=True)
    panel = prepare_signal_panel(factor, forward[["1D"]], groups=groups, quantiles=2, max_loss=0)
    original = panel.factor_weights()
    returned = panel.factor_weights()
    returned.replace_column(returned.get_column_index("weight"), returned["weight"] * 0)
    factor.iloc[:] = 0
    forward.iloc[:, :] = 0
    groups.iloc[:] = "changed"
    assert panel.factor_weights().equals(original)
    returns = panel.factor_returns()
    assert returns["date"].dtype == pl.Datetime("ns", "UTC")
    assert returns["date"].cast(pl.Int64).to_list() == [dates[1].value, dates[3].value]
    assert returns["weighted_return"].to_list() == pytest.approx([1 / 48, 1 / 48])


def test_explicit_factor_return_grid_preserves_holes_and_rejects_asset_expansion():
    factor, forward, groups, dates = _inputs()
    panel = prepare_signal_panel(factor, forward[["1D"]], groups=groups, quantiles=2, max_loss=0)
    grid = pd.DataFrame({"date": pd.date_range(dates[1], dates[3], freq="D")})
    before = grid.copy(deep=True)
    result = panel.factor_returns(date_grid=grid)
    assert result["n_obs"].to_list() == [4, 0, 0, 0, 4]
    assert result["weighted_return"].to_list() == [
        pytest.approx(1 / 48),
        None,
        None,
        None,
        pytest.approx(1 / 48),
    ]
    with pytest.raises(ValueError, match="only for aggregate"):
        panel.factor_returns(by_asset=True, date_grid=grid)
    pd.testing.assert_frame_equal(grid, before)


@pytest.mark.parametrize("flag", ["demeaned", "group_adjust", "equal_weight", "by_asset"])
def test_string_boolean_cannot_silently_change_factor_weight_or_contribution_policy(flag):
    factor, forward, _, _ = _inputs()
    panel = prepare_signal_panel(factor, forward[["1D"]], quantiles=2, max_loss=0)
    with pytest.raises(ValueError, match=flag):
        panel.factor_returns(**{flag: "false"})


@pytest.mark.parametrize("route", ["weights", "returns"])
@pytest.mark.parametrize("representation", ["pandas", "polars"])
@pytest.mark.parametrize("invalid", ["warmup", "infinite", "duplicate_batch"])
def test_direct_weight_boundary_rejects_produced_invalid_features_and_overlapping_batches(
    route, representation, invalid
):
    from ml4t.diagnostic.signal import compute_factor_returns, compute_factor_weights

    factor, forward, groups, _ = _inputs(zero_quote=invalid == "infinite")
    if invalid == "warmup":
        factor = factor.groupby(level="asset").transform(lambda values: values.rolling(20).mean())
    raw = factor.to_frame().join(forward[["1D"]]).join(groups).reset_index()
    if invalid == "duplicate_batch":
        raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    passed = pl.from_pandas(raw) if representation == "polars" else raw
    before = passed.clone() if representation == "polars" else passed.copy(deep=True)
    with pytest.raises(ValueError, match="duplicate|finite factors"):
        if route == "weights":
            compute_factor_weights(passed)
        else:
            compute_factor_returns(passed, return_columns=["1D"])
    if representation == "polars":
        assert passed.equals(before)
    else:
        pd.testing.assert_frame_equal(passed, before)


@pytest.mark.parametrize("demeaned", [False, True])
@pytest.mark.parametrize("equal_weight", [False, True])
def test_finite_feature_units_do_not_overflow_factor_weight_gross_normalization(
    demeaned, equal_weight
):
    from ml4t.diagnostic.signal import compute_factor_weights

    factor, _, groups, _ = _inputs()
    scaled = (factor * 1e308).to_frame().join(groups)
    result = compute_factor_weights(scaled, demeaned=demeaned, equal_weight=equal_weight)
    expected = _expected(demeaned, False, equal_weight, False, True) * 2
    assert result["weight"].to_list() == pytest.approx(expected, abs=1e-14)
