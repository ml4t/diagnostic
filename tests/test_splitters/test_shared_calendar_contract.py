"""Calendar selection must preserve declaration presence and actual row ownership."""

import warnings

import numpy as np
import pandas as pd
import pytest
from ml4t.specs import FeedSpec

from ml4t.diagnostic.core.purging import apply_purging_and_embargo, calculate_purge_indices
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


def test_config_serialization_filters_do_not_resurrect_an_explicitly_excluded_calendar():
    config = WalkForwardConfig(calendar=None)
    assert config.model_dump(include={"n_splits"}) == {"n_splits": 5}
    assert "calendar" not in config.model_dump(exclude={"calendar"})
    assert config.model_dump(exclude_none=True)["calendar"] is None


@pytest.mark.parametrize(
    ("mask", "disabled"), [(False, True), (None, True), ({}, True), (True, False), (..., False)]
)
def test_nested_exclusion_masks_do_not_silently_reenable_a_disabled_calendar(mask, disabled):
    config = WalkForwardConfig(calendar=None)
    wire = config.model_dump(exclude_none=True, exclude={"calendar": mask})
    if disabled:
        assert "calendar" in wire and wire["calendar"] is None
    else:
        assert "calendar" not in wire
    restored = WalkForwardCV(
        config=WalkForwardConfig(**wire),
        feed_spec=FeedSpec(time_contract_version=2, calendar="24/7"),
    )
    assert restored.config.calendar == (None if disabled else "24/7")


@pytest.fixture(params=[None, "UTC"], ids=["naive-rows", "aware-rows"])
def sparse_observation_times(request):
    times = pd.to_datetime(
        ["2025-07-01", "2025-07-02", "2025-07-03", "2025-07-07", "2025-07-08", "2025-07-15"]
    )
    return times.tz_localize(request.param) if request.param else times


@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_row_holdouts_do_not_require_timezones_or_claim_calendar_day_sizes(
    sparse_observation_times, direction
):
    frame = pd.DataFrame({"feature": range(6)}, index=sparse_observation_times)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cv = WalkForwardCV(
            config=WalkForwardConfig(
                calendar=None, test_period=2, n_splits=1, test_size=1, fold_direction=direction
            )
        )
        folds = [(train.tolist(), test.tolist()) for train, test in cv.split(frame)]
    assert not caught
    assert cv.test_indices_.tolist() == [4, 5]
    assert folds == ([([0], [1])] if direction == "forward" else [([0, 1, 2], [3])])


def test_direct_observation_purge_accepts_consistently_naive_or_aware_keys(
    sparse_observation_times,
):
    times = sparse_observation_times
    assert calculate_purge_indices(
        timestamps=times, test_start=times[3], test_end=times[4], label_horizon=1, calendar=None
    ) == [2]


def test_observation_embargo_keeps_row_units_across_a_missing_week(sparse_observation_times):
    times = sparse_observation_times
    cleaned = apply_purging_and_embargo(
        np.arange(6),
        test_start=times[3],
        test_end=times[4],
        label_horizon=1,
        embargo_size=2,
        timestamps=times,
        calendar=None,
    )
    assert cleaned.tolist() == [0, 1]


def test_row_splits_do_not_move_a_purge_boundary_to_an_earlier_duplicate_timestamp():
    frame = pd.DataFrame(
        {"feature": range(4)},
        index=pd.to_datetime(["2025-07-03", "2025-07-03", "2025-07-07", "2025-07-07"], utc=True),
    )
    cv = WalkForwardCV(
        n_splits=2, test_size=1, train_size=2, label_horizon=1, consecutive=True, calendar=None
    )
    assert [(train.tolist(), test.tolist()) for train, test in cv.split(frame)] == [
        ([0], [2]),
        ([0, 1], [3]),
    ]


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
