"""Legacy input clock conversions are explicit, not inferred execution grids."""

from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest
from pandas.tseries.offsets import BusinessDay, CustomBusinessDay, Day, MonthEnd


def _reference():
    reference = pytest.importorskip("alphalens.utils")
    if version("alphalens-reloaded") != "0.5.0":
        pytest.skip("this comparison pins corrected Alphalens0.5.0")
    return reference


def test_observed_calendar_and_utc_naive_conversion_keep_distinct_dst_instants():
    from ml4t.diagnostic.compat.alphalens import utils

    dates = pd.bdate_range("2025-07-01", periods=12, tz="America/New_York").drop(
        pd.Timestamp("2025-07-04", tz="America/New_York")
    )
    calendar = utils.infer_trading_calendar(dates[:4], dates)
    assert calendar.weekmask == "Mon Tue Wed Thu Fri"
    assert np.datetime64("2025-07-04") in calendar.holidays
    fold = pd.Timestamp("2025-11-02T01:30:00.000000001-04:00")
    assert utils.make_naive_ts(t=fold).value == fold.value
    assert utils.make_naive_ts(fold) == pd.Timestamp("2025-11-02T05:30:00.000000001")


@pytest.mark.parametrize(
    "frequency", [Day(), BusinessDay(), CustomBusinessDay(holidays=["2025-07-04"])]
)
def test_explicit_legacy_calendar_addition_preserves_nanoseconds_and_holiday_policy(frequency):
    from ml4t.diagnostic.compat.alphalens import utils

    start = pd.Timestamp("2025-07-03T10:00:00.000000001", tz="America/New_York")
    end = utils.add_custom_calendar_timedelta(
        input=start, timedelta=pd.Timedelta("1D1h1ns"), freq=frequency
    )
    date = "2025-07-07" if isinstance(frequency, CustomBusinessDay) else "2025-07-04"
    assert end == pd.Timestamp(f"{date}T11:00:00.000000002", tz="America/New_York")
    vector = utils.add_custom_calendar_timedelta(
        pd.DatetimeIndex([start]), pd.Timedelta("1D1h1ns"), frequency
    )
    assert vector[0] == end


def test_legacy_local_calendar_and_physical_elapsed_duration_are_distinct_through_dst():
    from ml4t.diagnostic.compat.alphalens import utils

    start = pd.Timestamp("2025-10-31T10:00:00.000000001", tz="America/New_York")
    end = utils.add_custom_calendar_timedelta(start, pd.Timedelta("1D1h1ns"), BusinessDay())
    assert end == pd.Timestamp("2025-11-03T11:00:00.000000002", tz="America/New_York")
    assert end - start == pd.Timedelta("3D2h1ns")
    assert utils.diff_custom_calendar_timedeltas(start, end, BusinessDay()) == pd.Timedelta(
        "1D2h1ns"
    )


def test_declared_day_offset_retains_the_pinned_pandas_dst_contract():
    from ml4t.diagnostic.compat.alphalens import utils

    start = pd.Timestamp("2025-11-01T12:00:00.000000001", tz="America/New_York")
    tick = utils.add_custom_calendar_timedelta(start, pd.Timedelta("1D"), Day())
    calendar_day = int(pd.__version__.split(".")[0]) >= 3
    expected_hour, elapsed_hours = (12, 25) if calendar_day else (11, 24)
    assert tick == pd.Timestamp(
        f"2025-11-02T{expected_hour}:00:00.000000001", tz="America/New_York"
    )
    assert tick - start == pd.Timedelta(elapsed_hours, unit="h")
    business = utils.add_custom_calendar_timedelta(start, pd.Timedelta("1D"), BusinessDay())
    assert business == pd.Timestamp("2025-11-03T12:00:00.000000001", tz="America/New_York")


@pytest.mark.parametrize("label", ["1ns", "2D3h4m5s6ms7us8ns", "-1ns", "0D"])
def test_legacy_duration_formatting_roundtrips_nanoseconds_without_day_coercion(label):
    from ml4t.diagnostic.compat.alphalens import utils

    value = pd.Timedelta(label)
    text = utils.timedelta_to_string(value)
    assert (pd.Timedelta(text) if text else pd.Timedelta(0)) == value
    assert utils.timedelta_strings_to_integers([label]) == [value.days]


def test_automatic_turnover_selection_does_not_turn_subday_horizons_into_zero_lags():
    from ml4t.diagnostic.compat.alphalens import utils

    columns = pd.Index(["factor", "30m", "1ns", "1D1h", "1D", "5D", "factor_quantile"])
    assert utils.get_forward_returns_columns(columns).tolist() == ["30m", "1ns", "1D1h", "1D", "5D"]
    assert utils.turnover_periods_from_columns(columns) == [1, 5]


@pytest.mark.parametrize("steps", [1, 2])
def test_backward_return_rekeying_uses_the_declared_date_clock_not_per_asset_compression(steps):
    from ml4t.diagnostic.compat.alphalens import utils

    dates = pd.date_range("2025-07-07T14:30Z", periods=5, freq="1ns")
    index = pd.MultiIndex.from_product([dates, [1001, 1002]], names=["date", "asset"])
    series = pd.Series(np.arange(10, dtype=float), index=index).drop((dates[2], 1002))
    before = series.copy(deep=True)
    result = utils.backshift_returns_series(series, steps)
    expected = {
        (dates[i - steps].value, asset): 2 * i + asset_index
        for i in range(steps, 5)
        for asset_index, asset in enumerate([1001, 1002])
        if (i, asset_index) != (2, 1)
    }
    assert {(date.value, asset): value for (date, asset), value in result.items()} == expected
    pd.testing.assert_series_equal(series, before)


def test_backshift_handles_zero_overlong_and_absent_first_target_without_fabricating_values():
    from ml4t.diagnostic.compat.alphalens import utils

    dates = pd.date_range("2025-07-07T14:30Z", periods=4, freq="1ns")
    series = pd.Series(
        [1.0, 2.0, 3.0, 4.0],
        index=pd.MultiIndex.from_product([dates, ["A"]], names=["date", "asset"]),
        name="returns",
    )
    pd.testing.assert_series_equal(utils.backshift_returns_series(series, 0), series)
    assert utils.backshift_returns_series(series, 10).empty
    assert utils.backshift_returns_series(series, 2**70).empty
    sparse = series.drop(dates[1], level="date")
    shifted = utils.backshift_returns_series(sparse, 1)
    assert shifted.tolist() == [3.0, 4.0]
    assert shifted.index.get_level_values(0).tolist() == [dates[1], dates[2]]


def test_common_clock_backshift_keeps_session_dates_as_civil_labels():
    from ml4t.diagnostic.compat.alphalens import utils

    dates = [pd.Timestamp(value).date() for value in ["2025-07-03", "2025-07-07"]]
    source = pd.Series(
        [0.0, 0.1], index=pd.MultiIndex.from_product([dates, ["A"]], names=["date", "asset"])
    )
    output = utils.backshift_returns_series(source, 1)
    assert output.index.get_level_values("date").tolist() == dates[:1]
    assert output.tolist() == [0.1]


def test_negative_reference_formatter_error_is_reconciled_against_exact_duration():
    reference = _reference()
    from ml4t.diagnostic.compat.alphalens import utils

    duration = pd.Timedelta(-1, unit="ns")
    old_text = reference.timedelta_to_string(duration)
    assert pd.Timedelta(old_text) != duration
    assert pd.Timedelta(utils.timedelta_to_string(duration)) == duration


@pytest.mark.parametrize(
    "frequency", [Day(), BusinessDay(), CustomBusinessDay(holidays=["2025-07-04"]), MonthEnd()]
)
def test_corrected_reference_duration_conventions_match_on_shared_dates(frequency):
    reference = _reference()
    from ml4t.diagnostic.compat.alphalens import utils

    start = pd.Timestamp("2025-07-03T14:30:00.000000001Z")
    end = pd.Timestamp("2025-09-01T15:30:00.000000003Z")
    actual = utils.diff_custom_calendar_timedeltas(start, end, frequency)
    assert actual == reference.diff_custom_calendar_timedeltas(start, end, frequency)
    assert utils.timedelta_to_string(actual) == reference.timedelta_to_string(actual)


def test_corrected_reference_backshift_matches_valid_sorted_legacy_input():
    reference = _reference()
    from ml4t.diagnostic.compat.alphalens import utils

    dates = pd.bdate_range("2025-07-07", periods=5, tz="UTC")
    source = pd.Series(
        np.arange(10, dtype=float),
        index=pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "asset"]),
    )
    pd.testing.assert_series_equal(
        utils.backshift_returns_series(source, 2), reference.backshift_returns_series(source, 2)
    )
