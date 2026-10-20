"""Legacy preparation calls must use the native common sample without rekeying."""

import warnings
from fractions import Fraction
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest


def _current_reference():
    reference = pytest.importorskip("alphalens.utils")
    if version("alphalens-reloaded") != "0.5.0":
        pytest.skip("these comparisons pin corrected Alphalens0.5.0, not the frozen profile")
    return reference


def _inputs(clock="30min"):
    if clock == "holiday":
        dates = pd.bdate_range("2025-07-01", periods=12, tz="America/New_York").drop(
            pd.Timestamp("2025-07-04", tz="America/New_York")
        )
    elif clock == "businessday":
        dates = pd.bdate_range("2025-07-07", periods=8, tz="UTC")
    elif clock == "overnight":
        dates = pd.date_range("2025-07-06T23:30Z", periods=8, freq="1h")
    elif clock == "dst":
        dates = pd.date_range("2025-11-02T00:30", periods=8, freq="1h", tz="America/New_York")
    else:
        dates = pd.date_range("2025-07-07T14:30Z", periods=8, freq=clock)
    assets = ["A", "B", "C", "D"]
    prices = pd.DataFrame(
        {
            asset: [100 + 10 * i + (i + 1) * day for day in range(len(dates))]
            for i, asset in enumerate(assets)
        },
        index=dates,
    )
    factor = pd.Series(
        np.tile([-3.0, -1.0, 1.0, 3.0], len(dates) - 2),
        index=pd.MultiIndex.from_product([dates[:-2], assets], names=["date", "asset"]),
        name="factor",
    )
    return factor, prices


@pytest.mark.parametrize(
    "clock,label",
    [
        ("30min", "30m"),
        ("1ns", "1ns"),
        ("holiday", "1D"),
        ("businessday", "1D"),
        ("1D", "1D"),
        ("overnight", "1h"),
        ("dst", "1h"),
    ],
)
@pytest.mark.parametrize("cumulative", [False, True])
def test_price_adapter_preserves_clock_endpoints_native_samples_and_exact_labels(
    clock, label, cumulative
):
    from ml4t.diagnostic.compat.alphalens import utils

    factor, prices = _inputs(clock)
    factor_before, prices_before = factor.copy(deep=True), prices.copy(deep=True)
    output = utils.compute_forward_returns(
        factor, prices, periods=(2, 1), cumulative_returns=cumulative
    )
    assert list(output) == [label, {"30m": "1h", "1ns": "2ns", "1D": "2D", "1h": "2h"}[label]]
    pd.testing.assert_index_equal(output.index, factor.index, check_names=True)
    assert output.iloc[0, 0] == pytest.approx(float(Fraction(1, 100)), rel=3e-16)
    expected_two = float(Fraction(2, 100) if cumulative else Fraction(1, 101))
    assert output.iloc[0, 1] == pytest.approx(expected_two, rel=3e-16)
    clean = utils.get_clean_factor(factor, output, quantiles=2, max_loss=0)
    direct = utils.get_clean_factor_and_forward_returns(
        factor,
        prices,
        periods=(1, 2),
        quantiles=2,
        max_loss=0,
        filter_zscore=None,
        cumulative_returns=cumulative,
    )
    pd.testing.assert_frame_equal(clean, direct)
    assert clean["factor_quantile"].tolist() == [1, 1, 2, 2] * (len(prices) - 2)
    assert clean.attrs["diagnostic"]["retained_rows"] == len(factor)
    pd.testing.assert_series_equal(factor, factor_before)
    pd.testing.assert_frame_equal(prices, prices_before)


def test_supplied_adapter_preserves_arbitrary_selectors_sparse_buckets_and_dated_groups():
    from ml4t.diagnostic.compat.alphalens import utils

    factor, _ = _inputs("1ns")
    outcomes = pd.DataFrame({"30m": 0.01, "1ns": -0.02, "1D1h": 0.03}, index=factor.index)
    groups = pd.Series([1, 1, 2, 2] * 6, index=factor.index)
    clean = utils.get_clean_factor(
        factor,
        outcomes,
        groupby=groups,
        groupby_labels={1: "first", 2: "second"},
        quantiles=None,
        bins=[-10, -5, 0, 5],
        max_loss=0,
    )
    assert list(clean) == ["30m", "1ns", "1D1h", "factor", "group", "factor_quantile"]
    assert clean["factor_quantile"].tolist() == [2, 2, 3, 3] * 6
    assert list(clean["group"].cat.categories) == ["first", "second"]
    pd.testing.assert_index_equal(clean.index, factor.index)


@pytest.mark.parametrize("identifiers", [[1001, 1002, 1003, 1004], [1, "1", ("C", 1), ("D", 1)]])
def test_adapter_does_not_stringify_asset_identity(identifiers):
    from ml4t.diagnostic.compat.alphalens import utils

    factor, prices = _inputs()
    dates = factor.index.get_level_values(0).unique()
    factor.index = pd.MultiIndex.from_product(
        [dates, pd.Index(identifiers, tupleize_cols=False)], names=["date", "asset"]
    )
    prices.columns = pd.Index(identifiers, tupleize_cols=False)
    output = utils.get_clean_factor_and_forward_returns(
        factor, prices, periods=(1,), quantiles=2, max_loss=0, filter_zscore=None
    )
    assert output.index.get_level_values("asset").tolist() == identifiers * len(dates)
    assert output["factor_quantile"].tolist() == [1, 1, 2, 2] * len(dates)


def test_adapter_exposes_loss_exception_and_rejects_timezone_conflicts_without_mutation():
    from ml4t.diagnostic.compat.alphalens import utils

    factor, prices = _inputs()
    short = prices.iloc[:2].copy()
    before = prices.copy(deep=True)
    with pytest.raises(utils.MaxLossExceededError, match="max_loss"):
        utils.get_clean_factor_and_forward_returns(
            factor, short, periods=(1,), quantiles=2, max_loss=0
        )
    mismatched = prices.tz_convert("America/New_York")
    with pytest.raises(utils.NonMatchingTimezoneError):
        utils.compute_forward_returns(factor, mismatched, periods=(1,))
    pd.testing.assert_frame_equal(prices, before)


def test_price_to_supplied_adapter_does_not_lose_explicit_calendar_none():
    from ml4t.diagnostic.compat.alphalens import utils

    factor, prices = _inputs()
    outcomes = utils.compute_forward_returns(factor, prices, periods=(1,), calendar=None)
    clean = utils.get_clean_factor(factor, outcomes, quantiles=2, max_loss=0)
    assert clean.attrs["diagnostic"]["calendar"] is None
    assert clean.attrs["diagnostic"]["calendar_source"] == "outcome_table"
    assert "calendar_argument" not in clean.attrs["diagnostic"]


def test_adapter_rejects_conflicting_declared_calendars_but_explicit_argument_wins():
    from ml4t.specs import FeedSpec

    from ml4t.diagnostic.compat.alphalens import utils

    factor, prices = _inputs()
    outcomes = utils.compute_forward_returns(factor, prices, periods=(1,), calendar=None)
    feed = FeedSpec(
        timestamp_col="date", entity_col="asset", time_contract_version=2, calendar="CME_Equity"
    )
    with pytest.raises(ValueError, match="calendar.*conflict"):
        utils.get_clean_factor(factor, outcomes, quantiles=2, max_loss=0, feed_spec=feed)
    clean = utils.get_clean_factor(
        factor, outcomes, quantiles=2, max_loss=0, feed_spec=feed, calendar="NYSE"
    )
    assert clean.attrs["diagnostic"]["calendar"] == "NYSE"
    assert clean.attrs["diagnostic"]["calendar_source"] == "argument"


def test_clean_adapter_metadata_is_owned_not_a_mutable_alias_of_its_producer_table():
    from ml4t.diagnostic.compat.alphalens import utils

    factor, prices = _inputs()
    outcomes = utils.compute_forward_returns(factor, prices, periods=(1,), calendar=None)
    clean = utils.get_clean_factor(factor, outcomes, quantiles=2, max_loss=0)
    clean.attrs["diagnostic"]["price_preparation"]["price_basis"] = "changed"
    assert outcomes.attrs["diagnostic"]["price_basis"] == "legacy_unspecified"


def test_preparation_profiles_are_explicit_and_do_not_change_native_return_definitions():
    from ml4t.diagnostic.compat.alphalens import utils

    factor, prices = _inputs("holiday")
    current = utils.get_clean_factor_and_forward_returns(
        factor, prices, periods=(1,), quantiles=2, max_loss=0
    )
    frozen = utils.get_clean_factor_and_forward_returns(
        factor, prices, periods=(1,), quantiles=2, max_loss=0, reference_version="0.4.6"
    )
    pd.testing.assert_frame_equal(current, frozen)
    assert current.attrs["diagnostic"]["reference_version"] == "0.5.0"
    assert frozen.attrs["diagnostic"]["reference_version"] == "0.4.6"
    assert frozen.attrs["diagnostic"]["price_preparation"]["reference_version"] == "0.4.6"
    with pytest.raises(ValueError, match="reference_version"):
        utils.get_clean_factor_and_forward_returns(factor, prices, reference_version="unknown")


def test_legacy_label_arithmetic_does_not_construct_deprecated_unitless_durations():
    from ml4t.diagnostic.compat.alphalens import utils

    factor, prices = _inputs("holiday")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", category=DeprecationWarning, module=r"ml4t\.diagnostic\.compat\.alphalens"
        )
        output = utils.compute_forward_returns(factor, prices, periods=(1, 2))
    assert output.columns.tolist() == ["1D", "2D"]
    np.testing.assert_allclose(output.iloc[0], [0.01, 0.02], rtol=3e-16, atol=0)


@pytest.mark.parametrize("clock", ["30min", "1ns", "overnight", "dst", "businessday", "1D"])
@pytest.mark.parametrize("cumulative", [False, True])
def test_reference_intraday_and_overnight_labels_masks_and_values_match(clock, cumulative):
    reference = _current_reference()
    from ml4t.diagnostic.compat.alphalens import utils

    factor, prices = _inputs(clock)
    factor_frequency = factor.index.levels[0].freq
    prices.iloc[2, 1] = np.nan
    expected = reference.compute_forward_returns(
        factor.copy(deep=True), prices, periods=(1, 2), cumulative_returns=cumulative
    )
    actual = utils.compute_forward_returns(
        factor, prices, periods=(1, 2), cumulative_returns=cumulative
    )
    pd.testing.assert_index_equal(actual.index, expected.index)
    assert actual.columns.tolist() == expected.columns.tolist()
    np.testing.assert_array_equal(actual.isna(), expected.isna())
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-16, equal_nan=True)
    assert factor.index.levels[0].freq == factor_frequency


@pytest.mark.parametrize("dated", [False, True])
@pytest.mark.parametrize(
    "buckets",
    [
        {"quantiles": 2},
        {"quantiles": [0, 0.25, 0.75, 1]},
        {"quantiles": None, "bins": 3},
        {"quantiles": None, "bins": [-10, -5, 0, 5]},
        {"quantiles": 4, "zero_aware": True},
        {"quantiles": 2, "binning_by_group": True},
    ],
)
def test_reference_bucket_options_and_static_or_dated_group_membership_match(dated, buckets):
    reference = _current_reference()
    from ml4t.diagnostic.compat.alphalens import utils

    factor, prices = _inputs("holiday")
    if dated:
        groups = pd.Series([1, 1, 2, 2] * (len(factor) // 4), index=factor.index)
        groups.iloc[4:8] = [2, 2, 1, 1]
    else:
        groups = {"A": 1, "B": 1, "C": 2, "D": 2}
    options = {
        "groupby": groups,
        "groupby_labels": {1: "first", 2: "second"},
        "periods": (1, 2),
        "filter_zscore": None,
        "max_loss": 0,
        **buckets,
    }
    expected = reference.get_clean_factor_and_forward_returns(factor, prices, **options)
    actual = utils.get_clean_factor_and_forward_returns(factor, prices, **options)
    pd.testing.assert_index_equal(actual.index, expected.index)
    np.testing.assert_array_equal(actual["factor_quantile"], expected["factor_quantile"])
    np.testing.assert_array_equal(actual["group"].astype(str), expected["group"].astype(str))
    np.testing.assert_allclose(
        actual[["1D", "2D", "factor"]], expected[["1D", "2D", "factor"]], rtol=2e-14, atol=2e-16
    )


@pytest.mark.parametrize("cumulative", [False, True])
@pytest.mark.parametrize("filter_zscore", [None, 0.5, 20])
def test_corrected_alphalens_reference_uses_identical_keys_values_masks_and_buckets(
    cumulative, filter_zscore
):
    reference = _current_reference()
    from ml4t.diagnostic.compat.alphalens import utils

    factor, prices = _inputs("holiday")
    prices.iloc[2, 1] = np.nan
    factor = factor.drop(factor.index[4])
    options = {
        "periods": (1, 2),
        "quantiles": 2,
        "max_loss": 1,
        "filter_zscore": filter_zscore,
        "cumulative_returns": cumulative,
    }
    expected = reference.get_clean_factor_and_forward_returns(factor, prices, **options)
    actual = utils.get_clean_factor_and_forward_returns(factor, prices, **options)
    pd.testing.assert_index_equal(actual.index, expected.index)
    assert actual.columns.tolist() == expected.columns.tolist()
    np.testing.assert_array_equal(actual["factor_quantile"], expected["factor_quantile"])
    np.testing.assert_allclose(actual[["1D", "2D"]], expected[["1D", "2D"]], rtol=2e-14, atol=2e-16)


def test_legacy_filter_uses_retained_date_levels_before_factor_rows_are_reindexed():
    reference = _current_reference()
    from ml4t.diagnostic.compat.alphalens import utils

    dates = pd.bdate_range("2025-07-07", periods=4, tz="UTC")
    prices = pd.DataFrame({"A": [100, 110, 220, 242], "B": [100] * 4}, index=dates)
    factor = pd.Series(
        [-1.0, 1.0] * 3,
        index=pd.MultiIndex.from_product([dates[:3], ["A", "B"]], names=["date", "asset"]),
    )
    factor = factor.drop(dates[1], level="date")
    assert len(factor.index.levels[0]) == 3 and factor.index.get_level_values(0).nunique() == 2
    expected = reference.compute_forward_returns(factor, prices, periods=(1,), filter_zscore=0.5)
    assert expected.loc[(slice(None), "A"), "1D"].isna().all()
    actual = utils.compute_forward_returns(factor, prices, periods=(1,), filter_zscore=0.5)
    pd.testing.assert_index_equal(actual.index, expected.index)
    np.testing.assert_array_equal(actual.isna(), expected.isna())
    np.testing.assert_allclose(actual, expected, equal_nan=True, rtol=2e-14, atol=2e-16)
    np.testing.assert_array_equal(actual.isna(), [[True], [False]] * 2)
    from ml4t.diagnostic.signal import prepare_signal_panel

    native = prepare_signal_panel(
        factor.rename("factor"),
        prices=prices,
        horizons={"one": 1},
        price_basis="raw",
        calendar=None,
        return_filter_zscore=0.5,
        quantiles=2,
        max_loss=0,
    )
    assert native.data.height == 4
    assert actual.attrs["diagnostic"]["filter_clock_rows"] == 3
    assert native.metadata["price_preparation"]["filter_clock_rows"] == 2
