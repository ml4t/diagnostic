"""Direct IC must correlate the selected unique observations, not joined shadows."""

import math

import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.metrics import (
    compute_forward_returns,
    cross_sectional_ic,
    cross_sectional_ic_series,
)


def _inputs(nanosecond=False):
    dates = pd.bdate_range("2025-01-02", periods=5, tz="UTC")
    if nanosecond:
        dates = pd.DatetimeIndex([dates[0] + pd.Timedelta(i * 100, unit="ns") for i in range(5)])
    wide = pd.DataFrame(
        {
            "A": [128.0, 144.0, 180.0, 202.5, 253.125],
            "B": [128.0, 160.0, 140.0, 175.0, 153.125],
            "C": [128.0, 192.0, 288.0, 432.0, 648.0],
            "D": [128.0, 256.0, 288.0, 576.0, 648.0],
        },
        index=dates,
    )
    features = wide.pct_change(fill_method=None).loc[[dates[1], dates[3]]].stack(future_stack=True)
    predictions = features.rename("prediction").rename_axis(index=["date", "asset"]).reset_index()
    marks = (
        wide.stack(future_stack=True)
        .rename("close")
        .rename_axis(index=["date", "asset"])
        .reset_index()
    )
    returns = compute_forward_returns(marks, periods=1, group_col="asset").rename(
        columns={"fwd_ret_1": "forward_return"}
    )
    return predictions, returns, dates[1]


@pytest.fixture(params=["pandas", "polars"])
def representation(request):
    return request.param


@pytest.fixture(params=["series", "summary"])
def route(request):
    return request.param


@pytest.fixture(params=["spearman", "pearson"])
def method(request):
    return request.param


def _run(predictions, returns, representation, route, method, **options):
    if representation == "polars":
        predictions, returns = pl.from_pandas(predictions), pl.from_pandas(returns)
        before = predictions.clone(), returns.clone()
    else:
        before = predictions.copy(deep=True), returns.copy(deep=True)
    operation = cross_sectional_ic_series if route == "series" else cross_sectional_ic
    try:
        return operation(predictions, returns, min_obs=2, method=method, **options)
    finally:
        if representation == "polars":
            assert predictions.equals(before[0]) and returns.equals(before[1])
        else:
            pd.testing.assert_frame_equal(predictions, before[0])
            pd.testing.assert_frame_equal(returns, before[1])


def _assert_expected(result, route, method):
    if method == "spearman":
        expected = 0.0  # ranks1,2,3,4 versus3,1,4,2 have zero centered cross-product
    else:
        x, y = [0.125, 0.25, 0.5, 1.0], [0.25, -0.125, 0.5, 0.125]
        dx = [v - math.fsum(x) / 4 for v in x]
        dy = [v - math.fsum(y) / 4 for v in y]
        expected = math.fsum(a * b for a, b in zip(dx, dy, strict=True)) / math.sqrt(
            math.fsum(v * v for v in dx) * math.fsum(v * v for v in dy)
        )
    if route == "series":
        assert len(result) == 2
        assert result["n_obs"].to_list() == [4, 4]
        assert result["ic"].to_list()[1] == pytest.approx(expected, rel=0.0, abs=1e-14)
        observed = result["ic"].to_list()[0]
    else:
        assert result["n_periods"] == 2
        observed = result["ic_mean"]
    assert observed == pytest.approx(expected, rel=0.0, abs=1e-14)


def test_causal_quote_pairs_keep_independent_ic_and_input_ownership(representation, route, method):
    pred, ret, _ = _inputs()
    _assert_expected(
        _run(pred, ret, representation, route, method, entity_col="asset"), route, method
    )


@pytest.mark.parametrize("side", ["predictions", "returns"])
@pytest.mark.parametrize("conflicting", [False, True])
def test_overlapping_batches_are_rejected_before_they_reweight_cross_sectional_ic(
    side, conflicting, representation, route, method
):
    pred, ret, day = _inputs()
    frame, column = (pred, "prediction") if side == "predictions" else (ret, "forward_return")
    batch = frame.loc[frame["date"] == day].iloc[:2].copy()
    if conflicting:
        batch[column] = batch[column] + 0.5
    overlapping = pd.concat([frame, batch])
    if side == "predictions":
        pred = overlapping
    else:
        ret = overlapping
    with pytest.raises(ValueError, match="duplicate"):
        _run(pred, ret, representation, route, method, entity_col="asset")


def test_missing_entity_declaration_cannot_create_cartesian_pairs(representation, route, method):
    pred, ret, _ = _inputs()
    with pytest.raises(ValueError, match="duplicate"):
        _run(pred, ret.drop(columns="asset"), representation, route, method)


@pytest.mark.parametrize("side", ["predictions", "returns"])
def test_feed_join_missing_asset_keys_are_rejected_before_silent_pair_loss(
    side, representation, route, method
):
    pred, ret, day = _inputs()
    frame = pred if side == "predictions" else ret
    frame.loc[(frame["date"] == day) & (frame["asset"] == "A"), "asset"] = None
    with pytest.raises(ValueError, match="nonmissing"):
        _run(pred, ret, representation, route, method, entity_col="asset")


@pytest.mark.parametrize("units", [("ns", "us"), ("ms", "ns"), ("us", "us")])
def test_custom_time_exports_align_equivalent_storage_units_without_rounding(
    units, representation, route, method
):
    pred, ret, _ = _inputs()
    pred = pred.rename(columns={"date": "timestamp"}).astype(
        {"timestamp": f"datetime64[{units[0]}, UTC]"}
    )
    ret = ret.rename(columns={"date": "timestamp"}).astype(
        {"timestamp": f"datetime64[{units[1]}, UTC]"}
    )
    _assert_expected(
        _run(pred, ret, representation, route, method, date_col="timestamp", entity_col="asset"),
        route,
        method,
    )


@pytest.mark.parametrize(
    "overlap", ["causal_previous_return", "same_value_name", "extra_prediction_in_returns"]
)
def test_explicit_value_selection_cannot_correlate_a_shadow_column_with_itself(
    overlap, representation, route, method
):
    pred, ret, _ = _inputs()
    options = {"entity_col": "asset"}
    if overlap == "causal_previous_return":
        pred["forward_return"] = pred["prediction"]
    elif overlap == "same_value_name":
        pred, ret = (
            pred.rename(columns={"prediction": "value"}),
            ret.rename(columns={"forward_return": "value"}),
        )
        options.update(pred_col="value", ret_col="value")
    else:
        ret["prediction"] = ret["close"]
    _assert_expected(_run(pred, ret, representation, route, method, **options), route, method)


def test_submicrosecond_quotes_are_distinct_observations_not_rounded_duplicates(
    representation, route, method
):
    pred, ret, _ = _inputs(nanosecond=True)
    _assert_expected(
        _run(pred, ret, representation, route, method, entity_col="asset"), route, method
    )


def test_composite_entity_keys_preserve_venue_specific_assets(representation, route, method):
    pred, ret, _ = _inputs()
    for frame in [pred, ret]:
        frame["venue"] = frame["asset"].map({"A": "X", "B": "Y", "C": "X", "D": "Y"})
        frame["asset"] = frame["asset"].map({"A": "a", "B": "a", "C": "b", "D": "b"})
    _assert_expected(
        _run(pred, ret, representation, route, method, entity_col=["venue", "asset"]), route, method
    )
