"""Calendar selection must preserve declaration presence and actual row ownership."""

import warnings

import pandas as pd
import pytest
from ml4t.specs import FeedSpec

from ml4t.diagnostic.core.purging import calculate_purge_indices
from ml4t.diagnostic.splitters import CombinatorialCV, WalkForwardConfig, WalkForwardCV


def test_explicit_none_keeps_actual_holiday_and_weekend_observations_in_row_folds():
    frame = pd.DataFrame(
        {"feature": [1.0, 2.0, 3.0, 4.0]},
        index=pd.to_datetime(["2025-07-04", "2025-07-05", "2025-07-06", "2025-07-07"]),
    )
    cv = WalkForwardCV(n_splits=2, test_size=1, train_size=2, consecutive=True, calendar=None)
    assert cv.calendar is None
    assert [(train.tolist(), test.tolist()) for train, test in cv.split(frame)] == [
        ([0, 1], [2]),
        ([0, 1, 2], [3]),
    ]


def test_integer_purge_without_calendar_means_rows_across_real_nyse_holiday_gap():
    times = pd.to_datetime(["2025-07-03T15:00Z", "2025-07-07T15:00Z", "2025-07-08T15:00Z"])
    with warnings.catch_warnings(record=True) as caught:
        indices = calculate_purge_indices(
            timestamps=times, test_start=times[1], test_end=times[2], label_horizon=1, calendar=None
        )
    assert indices == [0]
    assert not caught
    assert (
        calculate_purge_indices(
            timestamps=times,
            test_start=times[1],
            test_end=times[2],
            label_horizon=pd.Timedelta(days=1),
            calendar=None,
        )
        == []
    )


@pytest.mark.parametrize("alias", ["calendar", "calendar_id"])
def test_canonical_calendar_config_roundtrip_keeps_explicit_disable_and_legacy_alias(alias):
    config = WalkForwardConfig(**{alias: None})
    assert config.calendar is None and config.calendar_id is None
    wire = config.to_dict(mode="json")
    assert wire["calendar"] is None and "calendar_id" not in wire
    assert WalkForwardConfig(**wire).calendar is None


def test_conflicting_calendar_aliases_are_rejected_instead_of_silently_dropping_one():
    with pytest.raises(ValueError, match="calendar.*calendar_id"):
        WalkForwardConfig(calendar=None, calendar_id="NYSE")
    assert WalkForwardConfig(calendar=None, calendar_id=None).calendar is None


def test_shared_feed_precedence_retains_omission_explicit_null_and_caller_override():
    feed = FeedSpec(time_contract_version=2, calendar="24/7")
    assert WalkForwardCV(feed_spec=feed).calendar.config.exchange == "24/7"
    assert WalkForwardCV(feed_spec=feed, calendar=None).calendar is None
    assert WalkForwardCV(feed_spec=feed, calendar="NYSE").calendar.config.exchange == "NYSE"
    assert (
        WalkForwardCV(feed_spec=FeedSpec(time_contract_version=2)).calendar.config.exchange
        == "NYSE"
    )
    assert (
        WalkForwardCV(feed_spec=FeedSpec(time_contract_version=2, calendar=None)).calendar is None
    )
    assert WalkForwardCV(feed_spec=FeedSpec(calendar=None)).calendar.config.exchange == "NYSE"


def test_config_roundtrip_does_not_turn_omitted_calendar_into_a_metadata_override():
    config = WalkForwardConfig(n_splits=2)
    wire = config.to_dict(mode="json")
    assert "calendar" not in wire and "calendar_id" not in wire
    loaded = WalkForwardConfig(**wire)
    feed = FeedSpec(time_contract_version=2, calendar="24/7")
    assert WalkForwardCV(config=loaded, feed_spec=feed).calendar.config.exchange == "24/7"
    assert WalkForwardCV(config=loaded, feed_spec=feed, calendar=None).calendar is None
    assert config.calendar == "NYSE" and "calendar" not in config.model_fields_set


def test_explicit_calendar_can_override_config_without_mutating_caller_configuration():
    config = WalkForwardConfig(calendar="NYSE")
    cv = WalkForwardCV(config=config, calendar=None)
    assert cv.calendar is None and cv.config.calendar is None
    assert config.calendar == "NYSE"


def test_panel_purge_uses_each_assets_next_observation_across_holiday_and_other_assets():
    dates = pd.to_datetime(
        [
            "2025-07-01T15:00Z",
            "2025-07-02T15:00Z",
            "2025-07-03T15:00Z",
            "2025-07-07T15:00Z",
            "2025-07-08T15:00Z",
            "2025-07-09T15:00Z",
        ]
    )
    panel = pd.DataFrame(
        {"asset": ["A", "B", "C"] * 6, "feature": list(range(18))},
        index=dates.repeat(3),
    )
    cv = CombinatorialCV(n_groups=6, n_test_groups=1, label_horizon=1, isolate_groups=False)
    for train, test in cv.split(panel[["feature"]], groups=panel["asset"]):
        for asset in ["A", "B", "C"]:
            asset_rows = [row for row, name in enumerate(panel["asset"]) if name == asset]
            for row, next_observation in zip(asset_rows, asset_rows[1:]):
                if next_observation in test:
                    assert row not in train, (
                        f"{asset}: label endpoint at test row{next_observation} leaked from train row{row}"
                    )
