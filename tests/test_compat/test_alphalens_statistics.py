"""Familiar statistical tables must preserve samples, definitions and native defaults."""

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.compat import alphalens as al


def _factor_data(binary_outcomes=True):
    dates = pd.bdate_range("2025-01-27", periods=6, tz="UTC")
    index = pd.MultiIndex.from_product([dates, ["A", "B", "C", "D"]], names=["date", "asset"])
    factors = pd.Series(np.tile([1.0, 2.0, 3.0, 4.0], 6), index=index, name="factor")
    scales = np.array([1.0, -1.0, 2.0, -2.0, 3.0, -3.0])
    outcomes = (
        np.array([2.0, 4.0, 8.0, 6.0]) / 64
        if binary_outcomes
        else np.array([0.02, 0.04, 0.08, 0.06])
    )
    returns = (scales[:, None] * outcomes).ravel()
    frame = al.utils.get_clean_factor(
        factors,
        pd.DataFrame({"1D": returns, "5D": -2 * returns}, index=index),
        groupby={"A": "red", "B": "red", "C": "blue", "D": "blue"},
        quantiles=2,
        max_loss=0,
    )
    return frame, dates, scales


def test_public_cleaner_to_ic_preserves_independently_ranked_dates_and_group_options():
    frame, dates, scales = _factor_data()
    before = frame.copy(deep=True)
    actual = al.performance.factor_information_coefficient(frame)
    assert actual.index.equals(dates.rename("date"))
    np.testing.assert_allclose(actual["1D"], np.sign(scales) * 0.8, atol=1e-12)
    np.testing.assert_allclose(actual["5D"], -np.sign(scales) * 0.8, atol=1e-12)
    adjusted = al.performance.factor_information_coefficient(frame, group_adjust=True)
    np.testing.assert_allclose(adjusted, 0.0, atol=1e-12)
    grouped = al.performance.factor_information_coefficient(frame, by_group=True)
    np.testing.assert_allclose(grouped.xs("red", level="group")["1D"], np.sign(scales))
    np.testing.assert_allclose(grouped.xs("blue", level="group")["1D"], -np.sign(scales))
    pd.testing.assert_frame_equal(frame, before)


def test_original_overview_month_end_alias_survives_pandas_three_without_changing_month_membership():
    frame, _, _ = _factor_data()
    actual = al.performance.mean_information_coefficient(frame, by_time="M")
    assert actual.index.tolist() == [
        pd.Timestamp("2025-01-31", tz="UTC"),
        pd.Timestamp("2025-02-28", tz="UTC"),
    ]
    np.testing.assert_allclose(actual["1D"], [0.16, -0.8], atol=1e-12)
    np.testing.assert_allclose(actual["5D"], [-0.16, 0.8], atol=1e-12)


def test_public_quantile_means_and_spread_keep_unit_gross_and_frozen_raw_definitions_separate():
    frame, dates, scales = _factor_data()
    means, errors = al.performance.mean_return_by_quantile(frame, by_date=True, demeaned=False)
    np.testing.assert_allclose(means.xs(1, level="factor_quantile")["1D"], scales * 3 / 64)
    np.testing.assert_allclose(means.xs(2, level="factor_quantile")["1D"], scales * 7 / 64)
    np.testing.assert_allclose(errors["1D"], np.tile(np.abs(scales) / 64, 2))
    spread, uncertainty = al.performance.compute_mean_returns_spread(means, 2, 1, errors)
    assert spread.index.equals(dates.rename("date"))
    np.testing.assert_allclose(spread["1D"], scales * 2 / 64)
    np.testing.assert_allclose(uncertainty["1D"], np.abs(scales) / (64 * np.sqrt(2)))
    raw, raw_error = al.performance.compute_mean_returns_spread(
        means, 2, 1, errors, reference_version="0.4.6"
    )
    np.testing.assert_allclose(raw["1D"], scales * 4 / 64)
    np.testing.assert_allclose(raw_error["1D"], np.abs(scales) * np.sqrt(2) / 64)
    _, unavailable = al.performance.compute_mean_returns_spread(means, 2, 1)
    assert unavailable is None


def test_nonbinary_demeaning_keeps_real_float_rank_differences_instead_of_fabricating_ties():
    frame, _, _ = _factor_data(binary_outcomes=False)
    actual = al.performance.factor_information_coefficient(frame, group_adjust=True)
    np.testing.assert_allclose(actual["1D"].iloc[:2], [-0.4, 0.4], atol=1e-12)


def _incomplete_outcomes():
    frame, dates, _ = _factor_data()
    prices = pd.DataFrame(
        100 + np.arange(6)[:, None] * np.array([1, 2, 4, 3]),
        index=dates,
        columns=["A", "B", "C", "D"],
    )
    forward = al.utils.compute_forward_returns(
        frame["factor"], prices, periods=(1, 5), filter_zscore=None
    )
    observations = frame[["factor", "group"]].join(forward)
    observations["factor_quantile"] = al.utils.quantize_factor(observations, quantiles=2)
    return observations, dates


def test_unfinished_forward_return_producer_preserves_each_horizons_missingness():
    frame, dates = _incomplete_outcomes()
    means, errors = al.performance.mean_return_by_quantile(frame, by_date=True, demeaned=False)
    assert means.xs(dates[-1], level="date")["1D"].isna().all()
    assert means.loc[(1, dates[0]), "1D"] == pytest.approx(0.015)
    assert means.loc[(2, dates[0]), "1D"] == pytest.approx(0.035)
    assert means.loc[(1, dates[0]), "5D"] == pytest.approx(0.075)
    assert means.xs(1, level="factor_quantile")["5D"].iloc[1:].isna().all()
    coefficients = al.performance.factor_information_coefficient(frame)
    assert coefficients["1D"].iloc[-1:].isna().all()
    assert coefficients["5D"].iloc[1:].isna().all()
    assert errors.xs(dates[-1], level="date").isna().all().all()


def test_native_supplied_buckets_do_not_require_an_unused_factor_column_or_drop_known_outcomes():
    from ml4t.diagnostic.signal.quantile import compute_quantile_statistics

    frame, dates = _incomplete_outcomes()
    supplied = frame.drop(columns="factor")
    before = supplied.copy(deep=True)
    actual = (
        compute_quantile_statistics(
            supplied,
            return_columns=["1D", "5D"],
            quantile_col="factor_quantile",
            factor_col=None,
            by_date=True,
            demeaned=False,
        )
        .to_pandas()
        .set_index(["quantile", "date", "return_column"])
    )
    assert actual.loc[(1, dates[0], "1D"), "n_obs"] == 2
    assert actual.loc[(1, dates[-1], "1D"), "n_obs"] == 0
    assert actual.loc[(2, dates[0], "5D"), "mean"] == pytest.approx(0.175)
    assert actual.loc[(2, dates[1], "5D"), "n_obs"] == 0
    pd.testing.assert_frame_equal(supplied, before)


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("by_time", [None, "ME", "W"])
def test_familiar_mean_ic_uses_the_actual_reference_time_and_group_samples(grouped, by_time):
    reference = pytest.importorskip("alphalens.performance")
    frame, _, _ = _factor_data()
    expected = reference.mean_information_coefficient(frame, by_group=grouped, by_time=by_time)
    actual = al.performance.mean_information_coefficient(frame, by_group=grouped, by_time=by_time)
    if isinstance(expected, pd.Series):
        pd.testing.assert_series_equal(actual, expected)
    else:
        pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("binary_outcomes", [False, True])
@pytest.mark.parametrize("adjusted", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
def test_familiar_ic_matches_actual_reference_without_changing_input_membership(
    adjusted, grouped, binary_outcomes
):
    reference = pytest.importorskip("alphalens.performance")
    frame, _, _ = _factor_data(binary_outcomes=binary_outcomes)
    expected = reference.factor_information_coefficient(
        frame, group_adjust=adjusted, by_group=grouped
    )
    actual = al.performance.factor_information_coefficient(
        frame, group_adjust=adjusted, by_group=grouped
    )
    assert actual.index.to_list() == expected.index.to_list()
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("binary_outcomes", [False, True])
@pytest.mark.parametrize("by_date", [False, True])
@pytest.mark.parametrize("by_group", [False, True])
@pytest.mark.parametrize("demeaned", [False, True])
@pytest.mark.parametrize("group_adjust", [False, True])
def test_familiar_quantile_statistics_match_actual_date_weighted_reference(
    by_date, by_group, demeaned, group_adjust, binary_outcomes
):
    reference = pytest.importorskip("alphalens.performance")
    frame, _, _ = _factor_data(binary_outcomes=binary_outcomes)
    options = {
        "by_date": by_date,
        "by_group": by_group,
        "demeaned": demeaned,
        "group_adjust": group_adjust,
    }
    expected = reference.mean_return_by_quantile(frame, **options)
    actual = al.performance.mean_return_by_quantile(frame, **options)
    for left, right in zip(actual, expected, strict=True):
        assert left.index.to_list() == right.index.to_list()
        pd.testing.assert_frame_equal(left, right, atol=1e-12, rtol=1e-12)
