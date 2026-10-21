"""Dated prices exercise complete return paths, not just isolated scalar matches."""

import importlib

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.compat import empyrical as api


@pytest.fixture
def prices():
    return pd.DataFrame(
        {"strategy": [100.0, 101.0, 98.0, np.nan, 103.0], "benchmark": [50, 50, 52, 51, 53]},
        index=pd.date_range("2025-01-02", periods=5, tz="UTC", name="date"),
    )


def test_dated_price_workflow_retains_missing_adjacencies_and_initial_capital(prices):
    before = prices.copy(deep=True)
    # The ndarray route never pads missing prices, independently of pandas version.
    sample = pd.DataFrame(
        api.simple_returns(prices.to_numpy()), index=prices.index[1:], columns=prices.columns
    )
    expected_returns = [[0.01, 0], [-3 / 101, 2 / 50], [np.nan, -1 / 52], [np.nan, 2 / 51]]
    np.testing.assert_allclose(sample, expected_returns, atol=1e-15, equal_nan=True)
    wealth = api.cum_returns(sample, starting_value=100)
    np.testing.assert_allclose(wealth, [[101, 100], [98, 104], [98, 102], [98, 106]])
    np.testing.assert_allclose(api.cum_returns_final(sample), [-0.02, 0.06], atol=1e-15)
    drawdown = api.drawdown_series(sample)
    np.testing.assert_allclose(
        drawdown, [[0, 0], [-3 / 101, 0], [-3 / 101, -1 / 52], [-3 / 101, 0]], atol=1e-15
    )
    np.testing.assert_allclose(api.max_drawdown(sample), [-3 / 101, -1 / 52], atol=1e-15)
    for result in [wealth, drawdown]:
        assert result.index.equals(sample.index)
        assert result.columns.equals(prices.columns)
    assert api.cum_returns_final(sample).index.equals(prices.columns)
    assert api.max_drawdown(sample).index.equals(prices.columns)
    pd.testing.assert_frame_equal(prices, before)


@pytest.mark.parametrize("container", ["array", "series", "frame"])
def test_price_missingness_uses_actual_container_and_pandas_version_contract(prices, container):
    sample = prices.to_numpy() if container == "array" else prices
    if container == "series":
        sample = prices["strategy"]
    expected = np.array([[0.01, 0], [-3 / 101, 2 / 50], [np.nan, -1 / 52], [np.nan, 2 / 51]])
    if container != "array" and int(pd.__version__.split(".")[0]) < 3:
        expected[:, 0] = [0.01, -3 / 101, 0, 5 / 98]
    actual = api.simple_returns(sample)
    np.testing.assert_allclose(
        actual, expected[:, 0] if container == "series" else expected, atol=1e-15, equal_nan=True
    )
    if container != "array":
        assert actual.index.equals(prices.index[1:])
    if container == "series":
        assert actual.name == "strategy"


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_large_integer_price_ticks_survive_float_conversion_and_unsigned_declines(dtype):
    base = 2**60
    prices = np.array([[base, base + 3], [base + 1, base + 2], [base + 3, base]], dtype=dtype)
    expected = [[1 / base, -1 / (base + 3)], [2 / (base + 1), -2 / (base + 2)]]
    np.testing.assert_allclose(api.simple_returns(prices), expected, rtol=1e-15, atol=0)


@pytest.mark.parametrize("container", ["array", "series", "frame"])
@pytest.mark.parametrize("start", [0, 1, 100])
def test_compounding_paths_keep_independent_columns_and_series_metadata_contract(container, start):
    frame = pd.DataFrame(
        {"loss": [-0.5, np.nan, 1.0], "gain": [0.1, 0.0, -0.5]},
        index=pd.date_range("2025-02-03", periods=3, tz="UTC", name="date"),
    )
    sample = frame.to_numpy() if container == "array" else frame
    if container == "series":
        sample = frame["loss"]
    wealth = np.array([[0.5, 1.1], [0.5, 1.1], [1, 0.55]])
    expected = wealth - 1 if start == 0 else wealth * start
    expected = expected[:, 0] if container == "series" else expected
    actual = api.cum_returns(sample, starting_value=start)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(api.cum_returns_final(sample, starting_value=start), expected[-1])
    if container == "series":
        assert isinstance(actual, pd.Series) and actual.name is None
        assert actual.index.equals(sample.index)
    elif container == "frame":
        pd.testing.assert_index_equal(actual.columns, frame.columns)


@pytest.mark.parametrize("container", ["array", "series", "frame"])
def test_drawdown_omits_anchor_but_counts_initial_loss_and_total_loss(container):
    frame = pd.DataFrame({"loss": [-0.5, 1, -1, np.nan], "gain": [0.1, 0.1, 0.0, 0.0]})
    sample = frame.to_numpy() if container == "array" else frame
    if container == "series":
        sample = frame["loss"]
    expected = np.array([[-0.5, 0], [0, 0], [-1, 0], [-1, 0]])
    if container == "series":
        expected = expected[:, 0]
    actual = api.drawdown_series(sample)
    assert len(actual) == len(sample), "The initial anchor is not an extra observed return."
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    np.testing.assert_allclose(api.max_drawdown(sample), -1 if container == "series" else [-1, 0])


@pytest.mark.parametrize("name", ["cum_returns", "drawdown_series", "max_drawdown"])
@pytest.mark.parametrize("container", ["array", "series", "frame"])
def test_path_output_buffers_retain_function_specific_shape_and_labels(name, container):
    prices = pd.DataFrame({"a": [100.0, 99.0, 101.0], "b": [50.0, 52.0, 49.0]})
    frame = prices.pct_change(fill_method=None).iloc[1:]
    sample = frame.to_numpy() if container == "array" else frame
    if container == "series":
        sample = frame["a"]
    shape = sample.shape
    if name == "drawdown_series":
        shape = (len(sample) + 1,) + shape[1:]
    elif name == "max_drawdown":
        shape = shape[1:]
    out = np.full(shape, -999.0)
    function = getattr(api, name)
    expected = function(sample)
    actual = function(sample, out=out)
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    if name == "drawdown_series":
        np.testing.assert_array_equal(out[0], 0)
        assert np.shares_memory(actual, out)
        assert isinstance(actual, type(sample))
    elif name == "max_drawdown" and container == "series":
        assert isinstance(actual, float) and actual == out.item()
    else:
        assert actual is out


@pytest.mark.parametrize("name", ["cum_returns", "drawdown_series", "max_drawdown"])
@pytest.mark.parametrize("container", ["array", "frame", "split-frame"])
def test_path_buffers_cannot_mutate_original_float32_price_produced_returns(name, container):
    values = np.array([[100.0, 50.0], [101.0, 52.0], [99.0, 51.0]], dtype=np.float32)
    returns = pd.DataFrame(values).pct_change(fill_method=None).iloc[1:]
    if container == "split-frame":
        returns = pd.DataFrame({key: returns[key].to_numpy().copy() for key in returns})
    sample = returns.to_numpy() if container == "array" else returns
    before = np.asarray(sample).copy()
    if name == "drawdown_series":
        # An anchor-inclusive view from a larger input buffer overlaps the actual returns.
        storage = np.empty((3, 2), dtype=np.float32)
        storage[1:] = sample
        sample = storage[1:]
        out = storage
        before = sample.copy()
    else:
        raw = sample if container == "array" else sample[0].to_numpy(copy=False)
        if name == "cum_returns":
            if container != "array":
                sample = sample[0]
            out = raw
        else:
            out = raw[0] if container == "array" else raw
        before = np.asarray(sample).copy()
    with pytest.raises(ValueError, match="overlap"):
        getattr(api, name)(sample, out=out)
    np.testing.assert_array_equal(np.asarray(sample), before)


@pytest.mark.parametrize("container", ["array", "series", "frame"])
def test_empty_paths_keep_legacy_function_specific_return_shapes(container):
    sample = (
        np.empty((0, 2)) if container == "array" else pd.DataFrame(columns=["a", "b"], dtype=float)
    )
    if container == "series":
        sample = pd.Series([], dtype=float, name="a")
    assert api.cum_returns(sample).shape == sample.shape
    assert np.isnan(api.cum_returns_final(sample))
    drawdown = api.drawdown_series(sample)
    maximum = api.max_drawdown(sample)
    assert np.all(np.isnan(drawdown)) and np.all(np.isnan(maximum))
    assert np.shape(drawdown) == (() if container == "series" else (1, 2))
    assert np.shape(maximum) == (() if container == "series" else (2,))


@pytest.mark.parametrize(
    "name",
    ["simple_returns", "cum_returns", "cum_returns_final", "drawdown_series", "max_drawdown"],
)
@pytest.mark.parametrize("container", ["array", "series", "frame"])
def test_real_published_reference_paths_match_shapes_missingness_labels_and_values(
    name, container, prices
):
    reference = pytest.importorskip("empyrical.stats")
    metadata = importlib.import_module("importlib.metadata")
    if metadata.version("empyrical-reloaded") != "0.7.0":
        pytest.skip("The repaired ndarray drawdown contract is pinned to published 0.7.0.")
    sample = prices if name == "simple_returns" else prices.pct_change(fill_method=None).iloc[1:]
    if container == "array":
        sample = sample.to_numpy()
    elif container == "series":
        sample = sample["strategy"]
    actual = getattr(api, name)(sample)
    expected = getattr(reference, name)(sample)
    assert type(actual) is type(expected)
    assert np.shape(actual) == np.shape(expected)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15, equal_nan=True)
    if isinstance(expected, (pd.Series, pd.DataFrame)):
        pd.testing.assert_index_equal(actual.index, expected.index)
    if isinstance(expected, pd.DataFrame):
        pd.testing.assert_index_equal(actual.columns, expected.columns)
