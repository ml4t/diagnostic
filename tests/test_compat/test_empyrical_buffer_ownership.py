"""Copied numerical views must not conceal writes into original data, masks or rates."""

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.compat import empyrical as ep


def masked_returns(dtype, frame=False):
    data = np.array([0.01, 0.0, -0.01, 0.02], dtype=dtype)
    mask = np.array([False, True, False, False])
    series = pd.Series(pd.arrays.FloatingArray(data, mask, copy=False), name="signal", copy=False)
    values = pd.DataFrame({"signal": series, "hedge": -series}, copy=False) if frame else series
    data[0] = 0.03
    assert (values["signal"] if frame else values).iloc[0] == pytest.approx(0.03)
    data[0] = 0.01
    return values, data, mask


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("frame", [False, True])
@pytest.mark.parametrize(
    "name", ["annual_volatility", "sharpe_ratio", "sortino_ratio", "downside_risk"]
)
def test_real_nullable_return_pipeline_rejects_output_into_its_original_numeric_buffer(
    name, dtype, frame
):
    values, data, mask = masked_returns(dtype, frame)
    original, original_mask = values.copy(deep=True), mask.copy()
    output = data[:2] if frame else data[:1].reshape(())
    with pytest.raises(ValueError, match="overlap"):
        getattr(ep, name)(values, out=output)
    if frame:
        pd.testing.assert_frame_equal(values, original)
    else:
        pd.testing.assert_series_equal(values, original)
    np.testing.assert_array_equal(mask, original_mask)


def test_public_nullable_missingness_buffer_cannot_be_overwritten_through_a_float_view():
    data = np.array([0.01, 0.0, -0.01, 0.02, 0.01, -0.02, 0.03, 0.02])
    mask = np.array([False, True, False, False, False, False, False, False])
    values = pd.Series(pd.arrays.FloatingArray(data, mask, copy=False), copy=False)
    mask[0] = True
    assert pd.isna(values.iloc[0])
    mask[0] = False
    original, original_mask = values.copy(deep=True), mask.copy()
    with pytest.raises(ValueError, match="overlap"):
        ep.annual_volatility(values, out=mask.view(np.float64).reshape(()))
    pd.testing.assert_series_equal(values, original)
    np.testing.assert_array_equal(mask, original_mask)


@pytest.mark.parametrize("kind", ["integer", "boolean"])
def test_numeric_masked_constructors_protect_original_storage_across_dtype_views(kind):
    data = (
        np.array([-1, 0, 1, 0, 1, -1, 0, 1], dtype=np.int64)
        if kind == "integer"
        else np.array([False, True, False, True, True, False, True, True])
    )
    mask = np.array([False, True, False, False, False, False, False, False])
    constructor = pd.arrays.IntegerArray if kind == "integer" else pd.arrays.BooleanArray
    values = pd.Series(constructor(data, mask, copy=False), copy=False)
    previous = data[0]
    data[0] = 1
    assert values.iloc[0] == 1
    data[0] = previous
    original = values.copy(deep=True)
    with pytest.raises(ValueError, match="overlap"):
        ep.annual_volatility(values, out=data.view(np.float64)[:1].reshape(()))
    pd.testing.assert_series_equal(values, original)


@pytest.mark.parametrize(
    "name,keyword",
    [
        ("sharpe_ratio", "risk_free"),
        ("sortino_ratio", "required_return"),
        ("downside_risk", "required_return"),
    ],
)
def test_periodic_rate_input_is_not_hidden_by_the_excess_return_copy(name, keyword):
    values = np.array([0.01, -0.01, 0.02, 0.03])
    rates = np.array([0.001, 0.002, 0.001, 0.002])
    original = rates.copy()
    with pytest.raises(ValueError, match="overlap"):
        getattr(ep, name)(values, **{keyword: rates}, out=rates[:1].reshape(()))
    np.testing.assert_array_equal(rates, original)


def test_supplied_downside_estimate_cannot_double_as_the_output_buffer():
    values = np.array([[0.01, -0.02], [-0.01, 0.01], [0.02, 0.03]])
    deviation = np.array([0.1, 0.2])
    original = deviation.copy()
    with pytest.raises(ValueError, match="overlap"):
        ep.sortino_ratio(values, _downside_risk=deviation, out=deviation)
    np.testing.assert_array_equal(deviation, original)


@pytest.mark.parametrize("frame", [False, True])
def test_independent_nullable_output_preserves_values_and_existing_result_shape(frame):
    values, _, _ = masked_returns(np.float64, frame)
    original = values.copy(deep=True)
    output = np.full(2 if frame else (), -99.0)
    result = ep.annual_volatility(values, annualization=1, out=output)
    expected = np.std([0.01, -0.01, 0.02], ddof=1)
    np.testing.assert_allclose(result, [expected, expected] if frame else expected)
    if frame:
        assert result is output
        pd.testing.assert_frame_equal(values, original)
    else:
        pd.testing.assert_series_equal(values, original)


def test_uninspectable_extension_out_has_an_explicit_owned_snapshot_migration_path():
    values = pd.Series(pd.arrays.SparseArray([-0.01, 0.0, 0.01, 0.0], fill_value=0.0))
    original = values.copy(deep=True)
    output = np.full((), -99.0)
    with pytest.raises(TypeError, match="owned NumPy snapshot"):
        ep.annual_volatility(values, annualization=1, out=output)
    assert output.item() == -99
    dense = values.to_numpy(dtype=float, na_value=np.nan, copy=True)
    actual = ep.annual_volatility(dense, annualization=1, out=output)
    assert actual == pytest.approx(np.sqrt(0.0002 / 3))
    pd.testing.assert_series_equal(values, original)
