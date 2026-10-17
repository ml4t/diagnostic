"""Aligned levels must not become CAR or silently change event/date weights."""

from itertools import product

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml4t.diagnostic.signal import (
    AlignedEventResult,
    align_event_paths,
    analyze_signal_events,
    prepare_signal_panel,
)


def _constant():
    dates = pd.bdate_range("2025-01-02", periods=5, tz="UTC")
    prices = pd.DataFrame({"A": [10.0] * 5, "B": [100.0] * 5}, index=dates)
    factor = prices.loc[dates[[0, 2, 4]]].stack().rename("factor")
    factor.index.names = ["date", "asset"]
    return factor, prices


def test_event_levels_are_not_rebased_and_clipped_windows_do_not_invent_observations():
    factor, prices = _constant()
    result = analyze_signal_events(
        factor, prices, before=1, after=1, input_kind="levels", demeaned=False
    )
    np.testing.assert_allclose(result.statistics["mean"], [55.0, 55.0, 55.0])
    assert result.statistics["n_obs"].to_list() == [2, 3, 2]
    assert result.statistics["std"].to_list() == [0.0, 0.0, 0.0]
    assert result.paths.filter(pl.col("status") == "window_clipped").height == 4
    assert result.event_counts["n_events"].to_list() == [2, 2, 2]
    assert result.metadata["event_zero_rebased"] is False
    assert result.metadata["units"] == "supplied_levels"


def test_simple_returns_use_the_shared_full_clock_wealth_product_with_nan_as_flat():
    dates = pd.date_range("2025-01-01", periods=4)
    values = pd.DataFrame({"A": [0.1, np.nan, -1.0, 0.2]}, index=dates)
    events = pd.DataFrame({"date": dates[[1, 3]], "asset": ["A", "A"]})
    result = analyze_signal_events(
        events, values, input_kind="returns", before=1, after=1, demeaned=False
    )
    np.testing.assert_allclose(
        result.paths["raw_value"].to_numpy(), [1.1, 1.1, 0.0, 0.0, 0.0, np.nan], equal_nan=True
    )
    assert result.metadata["units"] == "wealth_starting_at_one"
    assert result.paths["status"][-1] == "window_clipped"


def test_date_averaging_is_not_asset_weighting_and_repeated_events_keep_their_ids():
    dates = pd.date_range("2025-01-01", periods=2)
    values = pd.DataFrame({"A": [10.0, 100.0], "B": [20.0, 200.0], "C": [30.0, 300.0]}, index=dates)
    events = pd.DataFrame({"date": [dates[0]] * 3 + [dates[1]], "asset": ["A", "B", "C", "A"]})
    kwargs = {"input_kind": "levels", "before": 0, "after": 0, "demeaned": False}
    equal_dates = analyze_signal_events(events, values, **kwargs)
    equal_events = analyze_signal_events(events, values, **kwargs, mean_by_date=False)
    assert equal_dates.statistics["mean"][0] == 60
    assert equal_events.statistics["mean"][0] == 40
    assert equal_dates.observations["n_assets"].to_list() == [3, 1]
    duplicated = pd.concat([events.iloc[:1], events.iloc[:2]], ignore_index=True)
    duplicate_result = analyze_signal_events(duplicated, values, **kwargs, mean_by_date=False)
    assert duplicate_result.paths["event_id"].n_unique() == 3
    assert duplicate_result.statistics["mean"][0] == pytest.approx(40 / 3)


def _prepared():
    dates = pd.bdate_range("2025-01-02", periods=8, tz="UTC")
    prices = pd.DataFrame(
        np.arange(100.0, 106.0)
        * np.exp(np.arange(8)[:, None] * np.array([0.01, 0.02, -0.01, 0.03, -0.02, 0.005])),
        index=dates,
        columns=list("ABCDEF"),
    )
    factor = prices.iloc[[0, 2, 4, 6]].stack().rename("factor")
    factor.index.names = ["date", "asset"]
    forward = prices.pct_change(fill_method=None).shift(-1).stack().rename("1D").to_frame()
    forward.index.names = ["date", "asset"]
    groups = pd.Series(
        [
            "small" if asset in ("AB" if day < dates[4] else "AC") else "large"
            for day, asset in factor.index
        ],
        index=factor.index,
        name="group",
    )
    panel = prepare_signal_panel(factor, forward, quantiles=2, groups=groups, max_loss=0)
    return panel, prices


@pytest.mark.parametrize("demeaned,group_adjust,by_group", list(product([False, True], repeat=3)))
def test_price_produced_panel_matches_actual_alphalens_event_options_and_intermediate_samples(
    demeaned, group_adjust, by_group
):
    reference = pytest.importorskip("alphalens.performance")
    panel, prices = _prepared()
    frame = (
        panel.data.to_pandas()
        .rename(columns={"quantile": "factor_quantile"})
        .set_index(["date", "asset"])
    )
    expected = reference.average_cumulative_return_by_quantile(
        frame, prices, 1, 2, demeaned=demeaned, group_adjust=group_adjust, by_group=by_group
    )
    result = analyze_signal_events(
        panel.data,
        prices,
        before=1,
        after=2,
        input_kind="levels",
        demeaned=demeaned,
        group_adjust=group_adjust,
        by_group=by_group,
    )
    for row in result.statistics.iter_rows(named=True):
        for statistic in ["mean", "std"]:
            key = (
                (row["quantile"], statistic, row["group"])
                if by_group
                else (row["quantile"], statistic)
            )
            wanted = expected.loc[key, row["offset"]]
            if pd.isna(wanted):
                assert row[statistic] is None
            else:
                assert row[statistic] == pytest.approx(wanted, rel=2e-12, abs=2e-12)
    event_frame = panel.data.to_pandas()
    for row in result.paths.iter_rows(named=True):
        source = row["source_date"]
        if source is None:
            assert row["value"] is None
            continue
        event_date = pd.Timestamp(row["event_date"])
        peers = event_frame[event_frame.date == event_date]
        if group_adjust:
            peers = peers[peers.group == row["group"]]
        benchmark = (
            prices.loc[source, peers.asset.tolist()].mean() if (demeaned or group_adjust) else 0.0
        )
        assert row["raw_value"] == prices.loc[source, row["asset"]]
        assert row["benchmark"] == pytest.approx(benchmark)
        assert row["value"] == pytest.approx(row["raw_value"] - benchmark)
    assert result.paths.height == panel.data.height * 4
    assert result.event_counts["n_events"].sum() == panel.data.height


def test_polars_long_values_and_json_preserve_nanoseconds_timezone_and_ownership():
    dates = pd.DatetimeIndex(
        [pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(i, "ns") for i in range(3)]
    )
    prices = pd.DataFrame({"A": [10.0, np.nan, 30.0]}, index=dates)
    events = pd.DataFrame({"date": dates, "asset": ["A"] * 3})
    original_prices, original_events = prices.copy(deep=True), events.copy(deep=True)
    long = pl.from_pandas(prices.rename_axis("time").reset_index()).unpivot(
        index="time", variable_name="symbol", value_name="level"
    )
    result = analyze_signal_events(
        pl.from_pandas(events).rename({"date": "time", "asset": "symbol"}),
        long,
        date_col="time",
        asset_col="symbol",
        value_col="level",
        before=1,
        after=1,
        input_kind="levels",
        demeaned=False,
    )
    restored = AlignedEventResult.from_json(result.to_json())
    for name in ["paths", "observations", "statistics", "event_counts"]:
        assert_frame_equal(getattr(result, name), getattr(restored, name))
    assert result.paths["event_date"].dtype == pl.Datetime("ns", "UTC")
    assert (
        result.paths["event_date"].cast(pl.Int64).unique().sort().to_list() == dates.asi8.tolist()
    )
    view = result.paths
    view[0, "value"] = 999.0
    assert result.paths["value"][0] is None
    metadata = result.metadata
    metadata["input_kind"] = "changed"
    assert result.metadata["input_kind"] == "levels"
    pd.testing.assert_frame_equal(prices, original_prices)
    pd.testing.assert_frame_equal(events, original_events)


@pytest.mark.parametrize(
    "setting",
    [
        {"before": -1},
        {"after": True},
        {"input_kind": "prices"},
        {"group_adjust": True},
        {"by_group": True},
        {"mean_by_date": 1},
    ],
)
def test_invalid_event_options_reject_before_mutating_caller_input(setting):
    events, prices = _constant()
    original_events, original_prices = events.copy(deep=True), prices.copy(deep=True)
    with pytest.raises((ValueError, TypeError)):
        analyze_signal_events(events, prices, **{"input_kind": "levels", **setting})
    pd.testing.assert_series_equal(events, original_events)
    pd.testing.assert_frame_equal(prices, original_prices)


def test_absent_clock_event_and_nan_levels_remain_unavailable_not_zero():
    events, prices = _constant()
    frame = events.rename("factor").reset_index().iloc[:1]
    frame["date"] = pd.Timestamp("2030-01-01", tz="UTC")
    paths = align_event_paths(frame, prices, input_kind="levels", before=1, after=1)
    assert paths["status"].to_list() == ["event_not_on_clock"] * 3
    assert paths["value"].null_count() == 3
    assert paths["source_date"].dtype == pl.Datetime("ns", "UTC")
