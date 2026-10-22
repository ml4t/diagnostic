"""Civil grouping preserves real labels and compounds only supplied observations."""

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.compat import empyrical as ep


def test_iso_week_crossing_keeps_one_week_without_inventing_holiday_rows():
    dates = pd.to_datetime(["2019-12-30", "2019-12-31", "2020-01-02", "2020-01-03"], utc=True)
    returns = pd.Series([0.1, 0.2, np.nan, -0.1], index=dates, name="strategy")
    original = returns.copy(deep=True)
    result = ep.aggregate_returns(returns, "weekly")
    assert result.index.to_list() == [(2020, 1)]
    assert result.iloc[0] == pytest.approx(1.1 * 1.2 * 0.9 - 1)
    pd.testing.assert_series_equal(returns, original)


@pytest.mark.parametrize("period", ["weekly", "monthly", "quarterly", "yearly"])
@pytest.mark.parametrize("frame", [False, True])
def test_actual_published_date_workflow_matches_keys_names_and_columnwise_products(period, frame):
    reference = pytest.importorskip("empyrical")
    dates = pd.to_datetime(["2019-12-30", "2020-01-02", "2020-03-31", "2020-04-01"]).tz_localize(
        "America/New_York"
    )
    dates.name = "observed_date"
    returns = pd.Series([0.1, np.nan, -0.1, 0.05], index=dates, name="strategy")
    if frame:
        returns = pd.DataFrame({"strategy": returns, "cash": np.zeros(4)})
    actual = ep.aggregate_returns(returns, period)
    expected = reference.aggregate_returns(returns.copy(deep=True), period)
    if frame:
        pd.testing.assert_frame_equal(actual, expected, check_exact=False, atol=1e-14, rtol=1e-14)
    else:
        pd.testing.assert_series_equal(actual, expected, check_exact=False, atol=1e-14, rtol=1e-14)


def test_civil_month_uses_declared_local_label_not_a_silent_utc_conversion():
    dates = pd.to_datetime(["2020-01-31 23:30", "2020-02-01 00:30"]).tz_localize("America/New_York")
    returns = pd.Series([0.1, 0.2], index=dates)
    result = ep.aggregate_returns(returns, "monthly")
    assert result.index.to_list() == [(2020, 1), (2020, 2)]
    np.testing.assert_allclose(result.to_numpy(), [0.1, 0.2])


def test_missing_entire_group_is_unchanged_wealth_not_a_missing_month_fill():
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-03-02"], utc=True)
    result = ep.aggregate_returns(pd.Series([np.nan, np.nan, 0.1], index=dates), "monthly")
    assert result.index.to_list() == [(2020, 1), (2020, 3)]
    np.testing.assert_allclose(result.to_numpy(), [0.0, 0.1])


def test_undated_returns_cannot_be_silently_treated_as_civil_dates():
    with pytest.raises((TypeError, ValueError), match="DatetimeIndex"):
        ep.aggregate_returns(pd.Series([0.1, 0.2]), "monthly")
