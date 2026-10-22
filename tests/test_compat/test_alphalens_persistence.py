"""Direct persistence calls must retain bucket clocks, signed shifts and actual ranks."""

from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.compat import alphalens as al


def _panel(profile="complete"):
    dates = pd.bdate_range("2025-03-06", periods=6, tz="UTC")
    index = pd.MultiIndex.from_product([dates, ["A", "B", "C"]], names=["date", "asset"])
    scores = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [1.0, 2.0, 3.0]] * 2)
    factors = pd.Series(scores.ravel(), index=index)
    outcomes = pd.DataFrame({"1D": np.tile([0.01, 0.02, 0.03], len(dates))}, index=index)
    if profile == "date_gap":
        factors = factors.drop(dates[2], level="date")
    data = al.utils.get_clean_factor(factors, outcomes, quantiles=2, max_loss=0)
    if profile == "bucket_gap":
        # A cleaner with fixed bin edges can produce dates without the selected bucket.
        factors.loc[dates[2]] = [4.0, 5.0, 6.0]
        data = al.utils.get_clean_factor(
            factors, outcomes, quantiles=None, bins=[0, 2.5, 10], max_loss=0
        )
    if profile == "observed":
        data.index = data.index.set_levels(
            pd.DatetimeIndex(data.index.levels[0], freq=None), level=0
        )
    return data, dates


def test_cleaner_to_turnover_and_rank_keeps_independent_membership_and_pearson_values():
    data, dates = _panel()
    before = data.copy(deep=True)
    membership = al.performance.quantile_turnover(data["factor_quantile"], 1)
    rank = al.performance.factor_rank_autocorrelation(data)
    assert membership.index.equals(dates.rename("date"))
    np.testing.assert_allclose(membership.iloc[1:3], [0.5, 0.5], atol=1e-12)
    np.testing.assert_allclose(rank.iloc[1:3], [-0.5, -0.5], atol=1e-12)
    assert pd.isna(membership.iloc[0]) and pd.isna(rank.iloc[0])
    pd.testing.assert_frame_equal(data, before)


def test_self_future_and_overlong_shifts_do_not_change_native_positive_lag_contract():
    from ml4t.diagnostic.signal import compute_quantile_turnover, compute_rank_autocorrelation

    data, _ = _panel()
    membership = al.performance.quantile_turnover(data["factor_quantile"], 1, period=0)
    rank = al.performance.factor_rank_autocorrelation(data, period=0)
    np.testing.assert_allclose(membership, 0)
    np.testing.assert_allclose(rank, 1)
    future = al.performance.factor_rank_autocorrelation(data, period=-1)
    assert future.iloc[0] == pytest.approx(-0.5) and pd.isna(future.iloc[-1])
    assert al.performance.quantile_turnover(data["factor_quantile"], 1, period=99).isna().all()
    for route in (compute_quantile_turnover, compute_rank_autocorrelation):
        with pytest.raises(ValueError, match="positive"):
            route(data.rename(columns={"factor_quantile": "quantile"}), lags=[0])


@pytest.mark.parametrize("profile", ["complete", "date_gap", "bucket_gap", "observed"])
@pytest.mark.parametrize("period", [-2, -1, 0, 1, 2, 9])
def test_actual_reference_persistence_keeps_bucket_specific_clock_masks_and_signed_lags(
    profile, period
):
    reference = pytest.importorskip("alphalens.performance")
    reference_version = version("alphalens-reloaded")
    if reference_version not in {"0.5.0", "0.4.6"}:
        pytest.skip("Frozen and corrected reference profiles are qualified independently")
    data, _ = _panel(profile)
    before = data.copy(deep=True)
    expected = reference.quantile_turnover(data["factor_quantile"], 1, period)
    actual = al.performance.quantile_turnover(
        data["factor_quantile"], 1, period, reference_version=reference_version
    )
    pd.testing.assert_series_equal(actual, expected, check_exact=False, atol=1e-12, rtol=1e-12)
    expected = reference.factor_rank_autocorrelation(data, period)
    actual = al.performance.factor_rank_autocorrelation(
        data, period, reference_version=reference_version
    )
    pd.testing.assert_series_equal(actual, expected, check_exact=False, atol=1e-12, rtol=1e-12)
    pd.testing.assert_frame_equal(data, before)
