"""Event studies must retain observed samples, elapsed bins and explicit rate units."""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml4t.diagnostic.signal import (
    SignalEventStudyResult,
    analyze_signal_event_study,
    prepare_signal_panel,
)


def _panel():
    dates = pd.to_datetime(["2025-01-02", "2025-01-06", "2025-01-08"], utc=True)
    index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "asset"])
    factor = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=index, name="factor")
    returns = pd.DataFrame({"3D": [0.03] * 6, "1D": [0.01] * 6}, index=index)
    return prepare_signal_panel(factor, returns, quantiles=1, max_loss=0)


def test_signal_event_study_uses_elapsed_bins_including_empty_bins_and_all_declared_events():
    result = analyze_signal_event_study(_panel(), rate_of_return=False, num_bars=3)
    assert result.event_distribution["n_events"].to_list() == [2, 0, 2, 2]
    assert result.event_distribution["date"].to_list() == list(
        pd.date_range("2025-01-02", "2025-01-08", freq="2D", tz="UTC")
    )
    assert result.metadata["distribution_unit"] == "elapsed"
    assert result.metadata["bin_width_ns"] == pd.Timedelta("2D").value
    stats = result.factor_statistics.row(0, named=True)
    assert stats["min"] == 1 and stats["max"] == 6 and stats["mean"] == 3.5
    assert stats["std"] == pytest.approx(np.sqrt(3.5))
    assert stats["count"] == 6 and stats["count_percent"] == 100
    assert result.event_returns is None
    assert result.metadata["event_paths_availability"] == "values_not_supplied"


def test_event_rate_conversion_uses_first_supplied_horizon_without_relabelling_raw_statistics():
    result = analyze_signal_event_study(_panel(), horizon_durations={"3D": "3D", "1D": "1D"})
    assert result.metadata["base_period"] == "3D"
    assert result.metadata["conversion_ratios"] == {"3D": 1, "1D": 3}
    one = result.mean_returns.filter(pl.col("return_column") == "1D").row(0, named=True)
    assert one["mean"] == pytest.approx(0.01)
    assert one["display_mean"] == pytest.approx(1.01**3 - 1)
    assert one["std"] == pytest.approx(0)
    assert one["sample_unit"] == "date" and one["n_obs"] == 3
    assert result.return_observations.filter(pl.col("return_column") == "1D")[
        "n_obs"
    ].to_list() == [2, 2, 2]
    with pytest.raises(ValueError, match="horizon_durations is required"):
        analyze_signal_event_study(_panel())


def test_event_study_serialization_owns_tables_and_keeps_undefined_rate_distinct_from_missing_mean():
    panel = _panel()
    frame = panel.data.to_pandas().set_index(["date", "asset"])
    factor = frame["factor"]
    returns = frame[["3D", "1D"]].copy()
    returns["1D"] = -1.5
    altered = prepare_signal_panel(factor, returns, quantiles=1, max_loss=0)
    result = analyze_signal_event_study(altered, horizon_durations={"3D": "1D", "1D": "2D"})
    undefined = result.mean_returns.filter(pl.col("return_column") == "1D")
    assert undefined["mean"][0] == -1.5 and undefined["display_mean"][0] is None
    assert undefined["display_status"][0] == "undefined_rate"
    restored = SignalEventStudyResult.from_json(result.to_json())
    assert_frame_equal(restored.mean_returns, result.mean_returns)
    copy = restored.factor_statistics
    copy[0, "mean"] = 999
    assert restored.factor_statistics["mean"][0] == 3.5


def test_single_date_event_distribution_is_one_observed_bin_not_a_zero_frequency_crash():
    frame = _panel().data.to_pandas()
    frame = frame[frame["date"] == frame["date"].iloc[0]].set_index(["date", "asset"])
    panel = prepare_signal_panel(frame["factor"], frame[["3D", "1D"]], quantiles=1, max_loss=0)
    result = analyze_signal_event_study(panel, rate_of_return=False)
    assert result.event_distribution["n_events"].to_list() == [2]
    assert result.metadata["bin_width_ns"] == 0


@pytest.mark.parametrize("num_bars", [True, 0, -1, 2.5])
def test_event_distribution_rejects_invalid_bin_requests(num_bars):
    with pytest.raises(ValueError, match="positive integer"):
        analyze_signal_event_study(_panel(), rate_of_return=False, num_bars=num_bars)
