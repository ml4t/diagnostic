"""Existing plots must consume the real dated persistence producer without empty legacy maps."""

import math

import numpy as np
import pandas as pd
import polars as pl
import pytest
from test_persistence_report_bridge import _analyze, _inputs

from ml4t.diagnostic.results.signal_results import TurnoverAnalysisResult
from ml4t.diagnostic.visualization.signal import plot_autocorrelation, plot_top_bottom_turnover
from ml4t.diagnostic.visualization.signal.turnover_plots import plot_turnover_heatmap

PROFILES = ["departures", "constant", "singleton", "single_date"]


def _component(profile, route, nanoseconds, restored):
    factor, marks, panel, dates = _inputs(nanoseconds, profile)
    result = _analyze(route, factor, marks, panel)
    component = result.to_turnover_result()
    if restored:
        component = type(component).model_validate_json(component.model_dump_json())
    return result, component, dates


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("route", ["raw", "prepared", "owned"])
@pytest.mark.parametrize("nanoseconds", [False, True])
@pytest.mark.parametrize("restored", [False, True])
def test_recorded_rank_dates_counts_and_missingness_reach_existing_plot(
    profile, route, nanoseconds, restored
):
    result, component, dates = _component(profile, route, nanoseconds, restored)
    before, original = component.model_dump_json(), result.to_json()
    figure = plot_autocorrelation(component)
    expected_dates = dates[[1]] if profile == "single_date" else dates[[1, 3]]
    assert [pd.Timestamp(value).value for value in figure.data[0].x] == [
        value.value for value in expected_dates
    ]
    values = [np.nan] if profile == "single_date" else [np.nan, 9 / math.sqrt(84)]
    if profile != "departures":
        values = [np.nan] * len(expected_dates)
    np.testing.assert_allclose(
        np.asarray(figure.data[0].y, dtype=float), values, rtol=0, atol=1e-14, equal_nan=True
    )
    assert figure.data[0].connectgaps is False
    assert figure.layout.meta["rank_scope"] == "date"
    assert figure.layout.meta["lag"] == 1 and figure.layout.meta["lag_unit"] == "observations"
    assert figure.layout.meta["correlation_method"] == "pearson"
    expected_counts = (
        [0]
        if profile == "single_date"
        else [0, 1 if profile == "singleton" else 3 if profile == "departures" else 4]
    )
    assert [row[1] for row in figure.data[0].customdata] == expected_counts
    assert all("95% CI" not in note.text for note in figure.layout.annotations)
    assert any("CI unavailable" in note.text for note in figure.layout.annotations)
    assert component.model_dump_json() == before and result.to_json() == original


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("route", ["raw", "prepared", "owned"])
@pytest.mark.parametrize("nanoseconds", [False, True])
@pytest.mark.parametrize("restored", [False, True])
def test_recorded_membership_turnover_dates_and_unavailable_estimates_reach_plot(
    profile, route, nanoseconds, restored
):
    result, component, dates = _component(profile, route, nanoseconds, restored)
    before, original = component.model_dump_json(), result.to_json()
    figure = plot_top_bottom_turnover(component)
    expected_dates = dates[[1]] if profile == "single_date" else dates[[1, 3]]
    assert len(figure.data) == (2 if profile in {"departures", "single_date"} else 1)
    for trace in figure.data:
        assert [pd.Timestamp(value).value for value in trace.x] == [
            value.value for value in expected_dates
        ]
        np.testing.assert_allclose(
            np.asarray(trace.y, dtype=float),
            [np.nan] if profile == "single_date" else [np.nan, 0.0],
            rtol=0,
            atol=0,
            equal_nan=True,
        )
        assert trace.connectgaps is False
        bucket = trace.meta["quantile"]
        initial = 1 if profile == "singleton" else 4 if profile == "constant" else 2
        current = 1 if profile == "departures" and bucket == 2 else initial
        assert [row[1] for row in trace.customdata] == (
            [initial] if profile == "single_date" else [initial, current]
        )
        assert [row[2] for row in trace.customdata] == (
            [None] if profile == "single_date" else [None, initial]
        )
    assert figure.layout.meta["turnover_method"] == "entrants_over_current"
    assert figure.layout.meta["lag"] == 1 and figure.layout.meta["lag_unit"] == "observations"
    assert component.model_dump_json() == before and result.to_json() == original


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("route", ["raw", "prepared", "owned"])
def test_turnover_heatmap_uses_actual_bucket_lag_means_not_empty_legacy_maps(profile, route):
    _, component, _ = _component(profile, route, False, False)
    figure = plot_turnover_heatmap(component)
    assert list(figure.data[0].x) == [1]
    buckets = [1, 2] if profile in {"departures", "single_date"} else [1]
    assert list(figure.data[0].y) == [f"Q{bucket}" for bucket in buckets]
    expected = [[np.nan] if profile == "single_date" else [0.0]] * len(buckets)
    np.testing.assert_allclose(
        np.asarray(figure.data[0].z, dtype=float), expected, rtol=0, atol=0, equal_nan=True
    )
    assert "Observations" in figure.layout.xaxis.title.text


def test_explicit_observation_lag_keeps_insufficient_history_missing_and_rejects_bad_selection():
    _, component, dates = _component("departures", "owned", True, True)
    _, _, panel, _ = _inputs(True)
    supplied = TurnoverAnalysisResult.from_observations(
        panel.quantile_turnover(lags=(1, 2)), panel.rank_autocorrelation(lags=(1, 2))
    )
    for function in [plot_autocorrelation, plot_top_bottom_turnover]:
        figure = function(supplied, lag=2)
        assert figure.layout.meta["lag"] == 2
        for trace in figure.data:
            assert [pd.Timestamp(value).value for value in trace.x] == [
                value.value for value in dates[[1, 3]]
            ]
            assert all(value is None or math.isnan(value) for value in trace.y)
        with pytest.raises(ValueError, match="lag"):
            function(component, lag=3)
        with pytest.raises(ValueError, match="lag"):
            function(component, lag=True)
    with pytest.raises(ValueError, match="period"):
        plot_autocorrelation(component, period="1D")
    with pytest.raises(ValueError, match="max_lags"):
        plot_autocorrelation(component, max_lags=1)
    figure = plot_autocorrelation(component, show_significance=False)
    assert not any("CI unavailable" in note.text for note in figure.layout.annotations)
    with pytest.raises(ValueError, match="lag"):
        plot_top_bottom_turnover(component, lag=2)


def test_plot_does_not_mutate_exported_dated_observations():
    _, component, _ = _component("departures", "owned", False, False)
    exported = component.get_dataframe("dated_turnover")
    exported.replace_column(
        exported.get_column_index("turnover"), pl.Series("turnover", [1.0] * exported.height)
    )
    figure = plot_top_bottom_turnover(component)
    assert all(trace.y[-1] == 0 for trace in figure.data)
