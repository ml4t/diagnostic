"""Price-produced columns retain per-metric counts, periodic rates and output contracts."""

import importlib
import math

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def api():
    return importlib.import_module("ml4t.diagnostic.compat.empyrical")


@pytest.fixture
def returns():
    prices = pd.DataFrame(
        {"complete": [100, 102, 101, 103, 104], "missing": [50, 51, 52, 51, np.nan]},
        index=pd.bdate_range("2025-01-02", periods=5, tz="UTC", name="date"),
    )
    return prices.pct_change(fill_method=None).iloc[1:]


def _expected(name, column, scale, rate=0.001):
    values = (
        [2 / 100, -1 / 102, 2 / 101, 1 / 103] if column == "complete" else [1 / 50, 1 / 51, -1 / 52]
    )
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    annual = (1.04 if column == "complete" else 1.02) ** (scale / 4) - 1
    if name in {"annual_return", "cagr"}:
        return annual
    if name == "annual_volatility":
        return std * math.sqrt(scale)
    if name == "calmar_ratio":
        return annual * (102 if column == "complete" else 52)
    if name == "sharpe_ratio":
        return (mean - rate) / std * math.sqrt(scale)
    downside = math.sqrt(sum(min(value - rate, 0) ** 2 for value in values) / len(values))
    return (
        downside * math.sqrt(scale)
        if name == "downside_risk"
        else (mean - rate) / downside * math.sqrt(scale)
    )


@pytest.mark.parametrize("scale", [1, 12, 52, 52.5, 252, 365])
@pytest.mark.parametrize(
    "name",
    [
        "annual_return",
        "cagr",
        "annual_volatility",
        "calmar_ratio",
        "sharpe_ratio",
        "sortino_ratio",
        "downside_risk",
    ],
)
def test_price_columns_use_metric_specific_missingness_and_explicit_periodic_rates(
    api, returns, name, scale
):
    before = returns.copy(deep=True)
    kwargs = {"period": "ignored-explicit-scale", "annualization": scale}
    if name == "sharpe_ratio":
        kwargs["risk_free"] = 0.001
    if name in {"sortino_ratio", "downside_risk"}:
        kwargs["required_return"] = 0.001
    function = getattr(api, name)
    expected = [_expected(name, column, scale) for column in ["complete", "missing"]]
    if name == "calmar_ratio":
        result = returns.apply(lambda column: function(column, **kwargs))
    else:
        result = function(returns, **kwargs)
        if name in {"annual_volatility", "sharpe_ratio"}:
            assert isinstance(result, np.ndarray)
        else:
            assert isinstance(result, pd.Series)
            assert list(result.index) == ["complete", "missing"]
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-12)
    for column, wanted in zip(["complete", "missing"], expected, strict=True):
        for sample in [returns[column], returns[column].to_numpy()]:
            actual = function(sample, **kwargs)
            assert np.ndim(actual) == 0
            np.testing.assert_allclose(actual, wanted, rtol=1e-10, atol=1e-12)
    pd.testing.assert_frame_equal(returns, before)


@pytest.mark.parametrize(
    "name", ["annual_volatility", "sharpe_ratio", "sortino_ratio", "downside_risk"]
)
def test_price_column_output_buffer_is_written_without_changing_input(api, returns, name):
    function = getattr(api, name)
    before = returns.copy(deep=True)
    out = np.empty(2)
    expected = function(returns.to_numpy(), annualization=12)
    actual = function(returns.to_numpy(), annualization=12, out=out)
    assert actual is out
    np.testing.assert_allclose(actual, expected)
    labeled = function(returns, annualization=12, out=out)
    assert np.shares_memory(labeled, out)
    if name in {"sortino_ratio", "downside_risk"}:
        assert isinstance(labeled, pd.Series)
        assert list(labeled.index) == ["complete", "missing"]
    scalar_out = np.empty(())
    scalar = function(returns["missing"], annualization=12, out=scalar_out)
    assert np.ndim(scalar) == 0 and scalar == scalar_out.item()
    invalid = np.full(3, -999.0)
    with pytest.raises(ValueError, match="shape"):
        function(returns, out=invalid)
    assert invalid.tolist() == [-999.0] * 3
    pd.testing.assert_frame_equal(returns, before)


def test_one_defined_price_return_retains_elapsed_rows_without_inventing_volatility(api, returns):
    sample = returns["missing"].where(np.arange(4) == 0)
    assert sample.notna().sum() == 1
    assert api.annual_return(sample, annualization=12) == pytest.approx(1.02**3 - 1)
    assert np.isnan(api.annual_volatility(sample))
    assert np.isnan(api.sharpe_ratio(sample))
    assert api.downside_risk(sample) == 0
    assert api.sortino_ratio(sample) == np.inf


@pytest.mark.parametrize("scale", [0, -1, np.nan, np.inf, True])
def test_invalid_explicit_annualization_is_rejected_instead_of_reinterpreted(api, scale):
    with pytest.raises(ValueError, match="positive.*finite"):
        api.annualization_factor("daily", scale)


def test_period_defaults_are_used_only_without_explicit_scale(api):
    assert [
        api.annualization_factor(period, None)
        for period in ["daily", "weekly", "monthly", "quarterly", "yearly"]
    ] == [252, 52, 12, 4, 1]
    assert api.annualization_factor("not-a-period", 52.5) == 52.5
    with pytest.raises(ValueError, match="period"):
        api.annualization_factor("not-a-period", None)
