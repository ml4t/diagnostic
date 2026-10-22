"""Cone redraws are explicit retrospective overlays, not forecasts trained on live data."""

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.compat import pyfolio as pf


def _inputs():
    returns = pd.Series(
        [-0.2, -0.2, 0.5, -0.3, 0.2], index=pd.date_range("2025-01-13", periods=5, tz="UTC")
    )
    bounds = pd.DataFrame(
        {
            1.0: [1.1] * 5,
            -1.0: [0.95, 0.9, 0.85, 0.8, 0.75],
            1.5: [1.15] * 5,
            -1.5: [0.925, 0.875, 0.825, 0.775, 0.725],
            2.0: [1.2] * 5,
            -2.0: [0.9, 0.85, 0.8, 0.75, 0.7],
        }
    )
    return returns, bounds


@pytest.mark.parametrize("strikes, starts", [(0, [0]), (1, [0, 0]), (2, [0, 0, 1]), (3, [0, 0, 1])])
def test_original_lower_two_std_crossing_selects_exact_redraw_membership(strikes, starts):
    returns, bounds = _inputs()
    result = pf.plotting.plot_cones("Account", bounds, returns, num_strikes=strikes)
    assert result.layout.meta["segment_start_rows"] == starts
    actual = next(trace for trace in result.data if trace.name.startswith("Actual"))
    np.testing.assert_allclose(actual.y, [0.8, 0.64, 0.96, 0.672, 0.8064])
    assert "retrospective" in result.layout.meta["description"]
    assert result.layout.title.text == "Account"
    assert result.layout.meta["num_samples"] == "unused by original supplied-bounds plotting route"


def test_redraw_bounds_shift_from_same_initial_forecast_without_resampling_or_mutating_inputs():
    returns, bounds = _inputs()
    before = bounds.copy(deep=True)
    figure = pf.plotting.plot_cones(None, bounds, returns, num_strikes=3, random_seed=5)
    lower = [trace for trace in figure.data if trace.name == "Cone 2 -2 std"][0]
    np.testing.assert_allclose(lower.y, np.array([0.9, 0.85, 0.8, 0.75]) - 0.36)
    assert list(lower.customdata) == returns.index[1:].astype(str).to_list()
    pd.testing.assert_frame_equal(bounds, before)
    with pytest.raises(TypeError, match="Plotly"):
        pf.plotting.plot_cones(None, bounds, returns, ax=object())


@pytest.mark.parametrize("strikes", [0, 1, 2, 3])
def test_actual_reference_artists_have_same_redraw_bounds_and_live_path(strikes):
    reference = pytest.importorskip("pyfolio.plotting")
    returns, bounds = _inputs()
    expected = reference.plot_cones(None, bounds, returns, num_strikes=strikes)
    actual = pf.plotting.plot_cones(None, bounds, returns, num_strikes=strikes)
    # Every filled reference band has one independently inspectable polygon.
    collections = expected.axes[0].collections
    native_bands = [trace for trace in actual.data if trace.fill == "tonexty"]
    assert len(native_bands) == len(collections)
    for polygon, upper in zip(collections, native_bands, strict=True):
        vertices = polygon.get_paths()[0].vertices
        lower = actual.data[list(actual.data).index(upper) - 1]
        n = len(upper.y)
        np.testing.assert_allclose(vertices[1 : n + 1, 1], upper.y)
        np.testing.assert_allclose(vertices[n + 2 : -1, 1][::-1], lower.y)
    np.testing.assert_allclose(expected.axes[0].lines[0].get_ydata(), actual.data[-1].y)
