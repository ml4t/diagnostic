"""Familiar cone calls must preserve the actual reference draws and financial definitions."""

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.compat import pyfolio as pf


def test_familiar_simulation_returns_samples_not_wealth_and_does_not_change_global_rng():
    source = pd.Series([0.1, -0.2, np.nan], index=pd.date_range("2025-01-01", periods=3))
    np.random.seed(876)
    before = np.random.get_state()
    result = pf.timeseries.simulate_paths(
        source, 5, starting_value=1000, num_samples=2, random_seed=42
    )
    np.testing.assert_allclose(
        result,
        [[np.nan, 0.1, np.nan, np.nan, 0.1], [0.1, np.nan, -0.2, np.nan, np.nan]],
        equal_nan=True,
    )
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])


def test_familiar_summary_treats_nan_as_neutral_and_zero_capital_as_net_return():
    samples = np.array([[0.1, np.nan, -1.0], [-0.1, 0.2, 0.0]])
    bounds = pf.timeseries.summarize_paths(samples, cone_std=(1, 2), starting_value=0)
    assert bounds.columns.to_list() == [1.0, -1.0, 2.0, -2.0]
    # Exact net-return paths are [.1,.1,-1] and [-.1,.08,.08].
    np.testing.assert_allclose(bounds[1.0], [0.1, 0.1, 0.08], atol=1e-15)
    np.testing.assert_allclose(bounds[-1.0], [-0.1, 0.08, -1.0], atol=1e-15)
    scaled = pf.timeseries.summarize_paths(samples, cone_std=1, starting_value=1000)
    np.testing.assert_allclose(scaled[1.0], (bounds[1.0] + 1) * 1000)
    assert bounds.index.equals(pd.RangeIndex(3))


def test_familiar_forecast_is_the_composition_with_the_same_draw_stream():
    source = pd.Series([0.1, -0.1, 0.0])
    expected = pf.timeseries.summarize_paths(
        pf.timeseries.simulate_paths(source, 7, num_samples=5, random_seed=4),
        cone_std=(0.5, 2),
        starting_value=10,
    )
    result = pf.timeseries.forecast_cone_bootstrap(
        source, 7, cone_std=(0.5, 2), starting_value=10, num_samples=5, random_seed=4
    )
    pd.testing.assert_frame_equal(result, expected)


def test_familiar_event_samples_preserve_missingness_name_and_partial_date_selection():
    index = pd.date_range("2025-01-31T12:30Z", periods=4)
    source = pd.Series([0.1, np.nan, -0.1, 0.2], index=index, name="strategy")
    before = source.copy(deep=True)
    result = pf.timeseries.extract_interesting_date_ranges(
        source, {"month": ("2025-02", "2025-02"), "absent": ("2023", "2023")}
    )
    assert list(result) == ["month"]
    expected = source.iloc[1:].copy()
    expected.index = expected.index.map(pd.Timestamp)
    pd.testing.assert_series_equal(result["month"], expected)
    pd.testing.assert_series_equal(source, before)
    assert not pf.timeseries.extract_interesting_date_ranges(source, {})


@pytest.mark.parametrize("starting_value", [0, 1, 1000, -5])
@pytest.mark.parametrize("samples,days", [(7, 9), (1, 1), (3, 0)])
def test_actual_published_pyfolio_draws_and_bounds_match_shared_inputs(
    starting_value, samples, days
):
    reference = pytest.importorskip("pyfolio.timeseries")
    source = pd.Series(
        [0.01, np.nan, -0.05, 0.03], index=pd.date_range("2025-01-13", periods=4, tz="UTC")
    )
    expected_draws = reference.simulate_paths(source, days, starting_value, samples, 47)
    actual_draws = pf.timeseries.simulate_paths(source, days, starting_value, samples, 47)
    np.testing.assert_array_equal(actual_draws, expected_draws)
    kwargs = {"cone_std": (0.5, 1.5, 2), "starting_value": starting_value}
    expected = reference.summarize_paths(expected_draws, **kwargs)
    actual = pf.timeseries.summarize_paths(actual_draws, **kwargs)
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-13, atol=1e-13)


def test_actual_published_pyfolio_custom_event_intermediates_match():
    reference = pytest.importorskip("pyfolio.timeseries")
    source = pd.Series(
        [0.01, np.nan, -0.02, 0.03],
        index=pd.date_range("2008-09-30T12:00Z", periods=4),
        name="strategy",
    )
    periods = {
        "partial": ("2008-09", "2008-10"),
        "instant": (source.index[1], source.index[2]),
        "unobserved": ("1990", "1990"),
    }
    expected = reference.extract_interesting_date_ranges(source, periods)
    actual = pf.timeseries.extract_interesting_date_ranges(source, periods)
    assert list(actual) == list(expected)
    for name in actual:
        pd.testing.assert_series_equal(actual[name], expected[name])
