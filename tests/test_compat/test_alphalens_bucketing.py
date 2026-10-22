"""Direct legacy bucket calls retain their own sample and error policy."""

from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest


def _input():
    from ml4t.diagnostic.metrics import compute_forward_returns

    dates = pd.bdate_range("2025-07-07", periods=3, tz="UTC")
    index = pd.MultiIndex.from_product([dates, list("ABCD")], names=["date", "asset"])
    prices = index.to_frame(index=False).assign(close=np.repeat([100.0, 101.0, 102.01], 4))
    forward = compute_forward_returns(prices, periods=1, group_col="asset").set_index(
        ["date", "asset"]
    )["fwd_ret_1"]
    assert forward.isna().sum() == 4
    return pd.DataFrame(
        {
            "factor": [-3.0, -1.0, 1.0, 3.0] * 3,
            "group": ["first", "first", "second", "second"] * 3,
            "1D": forward,
        },
        index=index,
    )


@pytest.mark.parametrize(
    ("options", "expected_ids"),
    [
        ({"quantiles": 2}, {1, 2}),
        ({"quantiles": [0, 0.25, 0.75, 1]}, {1, 2, 3}),
        ({"quantiles": np.array([0, 0.25, 0.75, 1])}, {1, 2, 3}),
        ({"quantiles": None, "bins": [-10, -5, 0, 5]}, {2, 3}),
        ({"quantiles": 4, "zero_aware": True}, {1, 2, 3, 4}),
        ({"quantiles": 2, "by_group": True}, {1, 2}),
    ],
)
def test_direct_bucketing_does_not_apply_future_outcome_missingness_or_rewrite_input(
    options, expected_ids
):
    from ml4t.diagnostic.compat.alphalens import utils

    data = _input()
    before = data.copy(deep=True)
    actual = utils.quantize_factor(data, **options)
    assert len(actual) == len(data)
    assert actual.name == "factor_quantile"
    assert set(actual) == expected_ids
    pd.testing.assert_frame_equal(data, before)


def test_bucket_failure_policy_is_per_date_and_legacy_cleaner_raises_value_error():
    from ml4t.diagnostic.compat.alphalens import utils

    data = _input()
    date = data.index.get_level_values(0)[4]
    data.loc[date, "factor"] = 1.0
    before = data.copy(deep=True)
    with pytest.raises(ValueError, match="[Bb]in edges|[Qq]uantile"):
        utils.get_clean_factor(data["factor"], data[["1D"]], quantiles=2, max_loss=0)
    with pytest.raises(ValueError, match="[Bb]in edges|[Qq]uantile"):
        utils.quantize_factor(data, quantiles=2)
    actual = utils.quantize_factor(data, quantiles=2, no_raise=True)
    assert len(actual) == 8
    assert date not in actual.index.get_level_values(0)
    pd.testing.assert_frame_equal(data, before)


def test_zero_aware_adapter_does_not_ignore_asymmetric_sign_buckets():
    from ml4t.diagnostic.compat.alphalens import utils

    date = pd.Timestamp("2025-07-07T14:30Z")
    data = pd.DataFrame(
        {"factor": [-4.0, -1.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0]},
        index=pd.MultiIndex.from_product([[date], list("ABCDEFGH")], names=["date", "asset"]),
    )
    ordinary = utils.quantize_factor(data, quantiles=4)
    signed = utils.quantize_factor(data, quantiles=4, zero_aware=True)
    assert ordinary.tolist() == [1, 1, 2, 2, 3, 3, 4, 4]
    assert signed.tolist() == [1, 2, 3, 3, 3, 4, 4, 4]


@pytest.mark.parametrize("by_group", [False, True])
@pytest.mark.parametrize("no_raise", [False, True])
def test_direct_bucket_reference_matches_values_keys_and_independent_no_raise_policy(
    by_group, no_raise
):
    reference = pytest.importorskip("alphalens.utils")
    if version("alphalens-reloaded") != "0.5.0":
        pytest.skip("this comparison pins corrected Alphalens0.5.0")
    from ml4t.diagnostic.compat.alphalens import utils

    data = _input()
    if no_raise:
        data.loc[data.index.get_level_values(0)[4], "factor"] = 1.0
    actual = utils.quantize_factor(data, quantiles=2, by_group=by_group, no_raise=no_raise)
    expected = reference.quantize_factor(data, quantiles=2, by_group=by_group, no_raise=no_raise)
    pd.testing.assert_series_equal(actual, expected)
