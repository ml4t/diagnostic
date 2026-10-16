"""Selected-factor wealth must use the declared sample before shared weighting."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

import ml4t.diagnostic.signal as signal
from ml4t.diagnostic.metrics import compute_forward_returns


def _inputs(nanoseconds=False, missing_tail=False):
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
    if missing_tail:
        prices.loc[dates[4]] = np.nan
    factor = (
        prices.pct_change(fill_method=None)
        .iloc[[1, 3]]
        .stack(future_stack=True)
        .rename("factor")
        .rename_axis(["date", "asset"])
    )
    marks = (
        prices.stack(future_stack=True).rename("close").rename_axis(["date", "asset"]).reset_index()
    )
    forward = (
        compute_forward_returns(marks, periods=[1], group_col="asset")
        .set_index(["date", "asset"])[["fwd_ret_1"]]
        .rename(columns={"fwd_ret_1": "1D"})
    )
    groups = pd.Series(
        [
            "L" if asset in (["A", "B"] if day == dates[1] else ["A", "C"]) else "H"
            for day, asset in factor.index
        ],
        index=factor.index,
        name="group",
    )
    return factor, forward, groups, dates


def _panel(nanoseconds=False):
    factor, forward, groups, dates = _inputs(nanoseconds)
    return signal.prepare_signal_panel(
        factor, forward, groups=groups, quantiles=2, max_loss=0
    ), dates


def _call(route, panel, **kwargs):
    if route == "owned":
        return panel.factor_cumulative_returns(return_column="1D", **kwargs)
    data = panel.data
    if route == "pandas":
        data = data.to_pandas().set_index(["date", "asset"])
    return signal.compute_factor_cumulative_returns(data, return_column="1D", **kwargs)


def _expected_weights(demeaned, group_adjust, equal_weight, second):
    if group_adjust:
        if demeaned:
            return [-0.25, -0.25, 0.25, 0.25] if second else [-0.25, 0.25, -0.25, 0.25]
        if equal_weight:
            return [0.25] * 4
        return [0.1, 0.1, 0.4, 0.4] if second else [1 / 6, 1 / 3, 1 / 6, 1 / 3]
    if equal_weight:
        return [-0.25, -0.25, 0.25, 0.25] if demeaned else [0.25] * 4
    return [-11 / 36, -7 / 36, 1 / 36, 17 / 36] if demeaned else [1 / 15, 2 / 15, 4 / 15, 8 / 15]


@pytest.mark.parametrize("route", ["owned", "pandas", "polars"])
@pytest.mark.parametrize("nanoseconds", [False, True])
@pytest.mark.parametrize("demeaned", [False, True])
@pytest.mark.parametrize("group_adjust", [False, True])
@pytest.mark.parametrize("equal_weight", [False, True])
def test_quote_factor_wealth_matches_independent_exposure_and_owns_exact_dates(
    route, nanoseconds, demeaned, group_adjust, equal_weight
):
    panel, dates = _panel(nanoseconds)
    before = panel.data
    actual = _call(
        route, panel, demeaned=demeaned, group_adjust=group_adjust, equal_weight=equal_weight
    )
    expected_returns = [
        sum(
            weight * outcome
            for weight, outcome in zip(
                _expected_weights(demeaned, group_adjust, equal_weight, second),
                [0.25, -0.125, 0.5, 0.125],
                strict=True,
            )
        )
        for second in [False, True]
    ]
    expected_wealth = [
        1 + expected_returns[0],
        (1 + expected_returns[0]) * (1 + expected_returns[1]),
    ]
    np.testing.assert_allclose(actual["weighted_return"], expected_returns, rtol=0, atol=1e-14)
    np.testing.assert_allclose(actual["wealth"], expected_wealth, rtol=0, atol=1e-14)
    assert [day.value for day in actual["date"].to_pandas()] == [day.value for day in dates[[1, 3]]]
    assert actual["n_obs"].to_list() == [4, 4]
    assert actual["compounding_clock"].to_list() == ["observed"] * 2
    assert actual["representation"].to_list() == ["wealth"] * 2
    assert actual["missing_return_policy"].to_list() == ["flat_wealth"] * 2
    actual.replace_column(actual.get_column_index("wealth"), pl.Series("wealth", [9.0, 9.0]))
    assert panel.data.equals(before)
    pd.testing.assert_index_equal(dates, _inputs(nanoseconds)[3])


@pytest.mark.parametrize("route", ["owned", "pandas", "polars"])
@pytest.mark.parametrize(
    "selectors,expected",
    [
        ({"quantiles": [1]}, [13 / 16, 169 / 256]),
        ({"quantiles": [2]}, [13 / 16, 169 / 256]),
        ({"groups": ["L"]}, [13 / 16, 117 / 128]),
        ({"groups": ["H"]}, [13 / 16, 117 / 128]),
        ({"quantiles": [1], "groups": ["L"]}, [13 / 16, 13 / 16]),
    ],
)
def test_bucket_and_dated_group_selection_precedes_normalization_not_wealth_filtering(
    route, selectors, expected
):
    panel, _ = _panel()
    actual = _call(route, panel, **selectors)
    np.testing.assert_allclose(actual["wealth"], expected, rtol=0, atol=1e-14)


@pytest.mark.parametrize("nanoseconds", [False, True])
def test_real_missing_forward_quotes_keep_unavailable_return_and_flat_not_fabricated_wealth(
    nanoseconds,
):
    factor, forward, groups, dates = _inputs(nanoseconds, missing_tail=True)
    data = factor.to_frame().join(forward).join(groups)
    actual = signal.compute_factor_cumulative_returns(data, return_column="1D")
    assert actual["weighted_return"].to_list()[1] is None
    assert actual["n_obs"].to_list() == [4, 0]
    np.testing.assert_allclose(actual["wealth"], [49 / 48, 49 / 48], rtol=0, atol=1e-14)
    assert [day.value for day in actual["date"].to_pandas()] == [day.value for day in dates[[1, 3]]]


def test_explicit_grid_retains_unscored_dates_without_calendar_inference_or_reweighting():
    panel, dates = _panel(True)
    clock = pl.DataFrame({"date": dates[[1, 2, 3]]})
    actual = _call("owned", panel, date_grid=clock)
    assert [day.value for day in actual["date"].to_pandas()] == [
        day.value for day in dates[[1, 2, 3]]
    ]
    assert actual["n_obs"].to_list() == [4, 0, 4]
    assert actual["weighted_return"][1] is None
    np.testing.assert_allclose(
        actual["wealth"], [49 / 48, 49 / 48, (49 / 48) ** 2], rtol=0, atol=1e-14
    )


def test_supplied_return_selector_is_not_parsed_as_duration_or_annualized():
    factor, forward, groups, dates = _inputs()
    panel = signal.prepare_signal_panel(
        factor, forward.rename(columns={"1D": "3h15m"}), groups=groups, quantiles=2, max_loss=0
    )
    actual = panel.factor_cumulative_returns(return_column="3h15m")
    assert actual["return_column"].to_list() == ["3h15m"] * 2
    np.testing.assert_allclose(actual["wealth"], [49 / 48, (49 / 48) ** 2], rtol=0, atol=1e-14)


@pytest.mark.parametrize("selectors", [{"quantiles": []}, {"groups": []}, {"groups": ["absent"]}])
def test_empty_real_selection_has_no_invented_dates_or_initial_capital_row(selectors):
    panel, _ = _panel()
    actual = _call("owned", panel, **selectors)
    assert actual.is_empty()
    assert actual["wealth"].dtype == pl.Float64


@pytest.mark.parametrize(
    "option,match",
    [
        ({"quantiles": "1"}, "sequence"),
        ({"quantiles": True}, "sequence"),
        ({"quantiles": [0]}, "positive integer"),
        ({"quantiles": [True]}, "positive integer"),
        ({"groups": "L"}, "sequence"),
        ({"demeaned": 1}, "boolean"),
        ({"group_adjust": "yes"}, "boolean"),
        ({"equal_weight": None}, "boolean"),
    ],
)
def test_invalid_public_selection_and_weight_options_fail_before_returning_wealth(option, match):
    panel, _ = _panel()
    with pytest.raises(ValueError, match=match):
        _call("owned", panel, **option)


@pytest.mark.parametrize("column", ["factor", "missing", "", None])
def test_owned_panel_cannot_compound_factor_or_unknown_selector_as_supplied_outcome(column):
    panel, _ = _panel()
    with pytest.raises(ValueError, match="return_column"):
        panel.factor_cumulative_returns(return_column=column)
