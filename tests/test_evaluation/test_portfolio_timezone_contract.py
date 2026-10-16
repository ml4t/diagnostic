"""Timezone adapters must not reject an unchanged labeled observation grid."""

from datetime import UTC, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.evaluation import PortfolioAnalysis


@pytest.mark.parametrize("zone", ["UTC", UTC, ZoneInfo("UTC")])
@pytest.mark.parametrize("date_adapter", ["inferred", "pandas", "polars"])
def test_equivalent_utc_objects_cannot_reject_valid_dates_or_mispair_reference_returns(
    zone, date_adapter
):
    index = pd.bdate_range("2025-01-02T00:00:00.000000001", periods=4, tz=zone)
    benchmark = pd.Series([0.01, -0.02, 0.03, 0.015], index=index)
    returns = 2 * benchmark
    dates = {"inferred": None, "pandas": index.copy(), "polars": pl.from_pandas(index)}[
        date_adapter
    ]
    analysis = PortfolioAnalysis(returns, benchmark.iloc[::-1], dates=dates)
    actual = pd.DatetimeIndex(analysis.dates.to_pandas())
    assert [stamp.value for stamp in actual] == [stamp.value for stamp in index]
    assert str(actual.tz) == "UTC"
    np.testing.assert_array_equal(analysis.returns, returns.to_numpy())
    np.testing.assert_array_equal(analysis.benchmark, benchmark.to_numpy())
    assert analysis.compute_summary_stats().beta == pytest.approx(2)


@pytest.mark.parametrize("conflict", ["naive", "different_zone", "different_instant"])
def test_normalizing_utc_objects_does_not_hide_conflicting_declared_dates(conflict):
    index = pd.bdate_range("2025-01-02T00:00:00.000000001", periods=4, tz=UTC)
    returns = pd.Series([0.01, -0.02, 0.03, 0.015], index=index)
    dates = {
        "naive": index.tz_localize(None),
        "different_zone": index.tz_convert("America/New_York"),
        "different_instant": index + pd.Timedelta(1, "ns"),
    }[conflict]
    with pytest.raises(ValueError, match="dates must agree"):
        PortfolioAnalysis(returns, dates=dates)


def test_explicit_fixed_offset_to_utc_conversion_preserves_each_instant():
    index = pd.bdate_range(
        "2025-01-02T00:00:00.000000001",
        periods=4,
        tz=timezone(timedelta(hours=5, minutes=30)),
    )
    returns = pd.Series([0.01, -0.02, 0.03, 0.015], index=index)
    converted = returns.tz_convert("UTC")
    analysis = PortfolioAnalysis(converted)
    assert [pd.Timestamp(stamp).value for stamp in analysis.dates.to_pandas()] == [
        stamp.value for stamp in index
    ]
