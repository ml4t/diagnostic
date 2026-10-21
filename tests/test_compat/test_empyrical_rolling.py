"""Row-window adapters preserve endpoints, calculations and caller-owned storage."""

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.compat import empyrical as ep

UNARY = [
    "max_drawdown",
    "cagr",
    "annual_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "downside_risk",
]
BINARY = [
    "excess_sharpe",
    "alpha_beta",
    "alpha_beta_aligned",
    "alpha",
    "alpha_aligned",
    "beta",
    "beta_aligned",
    "up_capture",
    "down_capture",
    "up_down_capture",
]


def sample(dated=False):
    benchmark = np.array([-0.02, -0.01, 0.0, 0.01, 0.02, -0.03])
    returns = 0.0003 + 1.2 * benchmark
    returns[2] = np.nan
    if dated:
        dates = pd.to_datetime(
            ["2025-01-02", "2025-01-03", "2025-01-03", "2025-01-07", "2025-01-10", "2025-01-13"],
            utc=True,
        )
        return pd.Series(returns, index=dates), pd.Series(benchmark, index=dates)
    return returns, benchmark


@pytest.mark.parametrize("name", [name for name in UNARY + BINARY if name != "cagr"])
@pytest.mark.parametrize("window", [1, 3, 6, 7])
@pytest.mark.parametrize("dated", [False, True])
def test_published_rolling_workflows_match_observed_row_endpoints_and_shapes(name, window, dated):
    reference = pytest.importorskip("empyrical.stats")
    returns, benchmark = sample(dated)
    args = (returns, benchmark) if name in BINARY else (returns,)
    actual = getattr(ep, "roll_" + name)(*args, window=window)
    expected = getattr(reference, "roll_" + name)(*args, window=window)
    assert np.shape(actual) == np.shape(expected)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11, equal_nan=True)
    if dated:
        assert actual.index.equals(expected.index)
        assert type(actual) is type(expected)


@pytest.mark.parametrize("window", [1, 3, 6, 7])
def test_rolling_cagr_annualizes_exact_products_instead_of_relabeling_window_returns(window):
    returns = np.array([0.1, -0.1, np.nan, -1.0, 0.2, 0.1])
    width = min(window, len(returns))
    expected = []
    for end in range(width, len(returns) + 1):
        product = 1.0
        for value in returns[end - width : end]:
            if not np.isnan(value):
                product *= 1 + value
        expected.append(product ** (12 / width) - 1)
    np.testing.assert_allclose(
        ep.roll_cagr(returns, window=window, annualization=12), expected, atol=1e-12, rtol=1e-12
    )


def test_roll_cagr_reference_error_is_reconciled_with_an_independent_window_oracle():
    reference = pytest.importorskip("empyrical.stats")
    with pytest.raises(TypeError, match="out"):
        reference.roll_cagr(np.array([0.1, 0.2, 0.3]), window=2)
    np.testing.assert_allclose(
        ep.roll_cagr(np.array([0.1, 0.2, 0.3]), window=2, annualization=2), [0.32, 0.56]
    )


@pytest.mark.parametrize("name", UNARY + BINARY)
def test_empty_rolling_input_remains_empty_without_inventing_a_warmup_observation(name):
    args = (np.array([]), np.array([])) if name in BINARY else (np.array([]),)
    assert getattr(ep, "roll_" + name)(*args, window=3).shape == (0,)


@pytest.mark.parametrize("window", [0, -1, 1.5, True])
def test_invalid_rolling_window_is_rejected_before_buffer_or_input_changes(window):
    values = np.array([0.1, -0.1, 0.2])
    original = values.copy()
    output = np.full(3, -99.0)
    with pytest.raises((TypeError, ValueError), match="positive integer"):
        ep.roll_sharpe_ratio(values, window=window, out=output)
    np.testing.assert_array_equal(values, original)
    np.testing.assert_array_equal(output, [-99.0] * 3)


def test_rolling_output_alias_is_checked_against_original_returns_and_benchmark():
    returns, benchmark = sample()
    for storage in (returns, benchmark):
        original = storage.copy()
        with pytest.raises(ValueError, match="overlap"):
            ep.roll_beta(returns, benchmark, window=3, out=storage[:4])
        np.testing.assert_array_equal(storage, original)


def test_rolling_buffer_identity_and_native_padded_output_contract_remain_distinct():
    from ml4t.diagnostic.evaluation.portfolio_analysis import PortfolioAnalysis

    values = np.array([0.1, -0.1, 0.2, 0.05])
    output = np.full(2, -99.0)
    assert ep.roll_sharpe_ratio(values, window=3, out=output) is output
    expected = [
        np.mean(values[:3]) / np.std(values[:3], ddof=1) * np.sqrt(252),
        np.mean(values[1:]) / np.std(values[1:], ddof=1) * np.sqrt(252),
    ]
    np.testing.assert_allclose(output, expected)
    native = PortfolioAnalysis._rolling_sharpe(values, 3, 0.0, 252)
    assert len(native) == 4 and np.isnan(native[:2]).all()
    np.testing.assert_allclose(native[2:], output)


def test_generic_callbacks_and_sign_helpers_forward_kwargs_without_fake_full_windows():
    values = np.array([0.1, -0.1, 0.2, 0.3])
    benchmark = np.array([0.01, -0.02, 0.02, 0.0])
    calls = []

    def callback(returns, factor, *, multiplier):
        calls.append((returns.copy(), factor.copy()))
        return float(np.sum(returns - factor) * multiplier)

    np.testing.assert_allclose(
        ep.roll(values, benchmark, window=3, function=callback, multiplier=2), [0.38, 0.8]
    )
    assert len(calls) == 2
    assert ep.roll(values, benchmark, window=5, function=callback, multiplier=2).size == 0
    assert ep.up(values, benchmark, function=callback, multiplier=2) == pytest.approx(0.54)
    assert ep.down(values, benchmark, function=callback, multiplier=2) == pytest.approx(-0.16)


def test_row_window_view_is_readonly_unless_mutability_is_explicitly_requested():
    values = np.arange(8.0).reshape(4, 2)
    windows = ep.rolling_window(values, 2)
    assert windows.shape == (3, 2, 2) and np.shares_memory(values, windows)
    with pytest.raises(ValueError):
        windows[0, 0, 0] = -1
    mutable = ep.rolling_window(values, 2, mutable=True)
    mutable[1, 0, 0] = -1
    assert values[1, 0] == -1 and mutable[0, 1, 0] == -1


def test_deprecated_downsize_alias_retains_the_same_shared_calculation():
    returns, _ = sample()
    with pytest.warns(DeprecationWarning, match="roll_downside_risk"):
        result = ep.roll_downsize_risk(returns, window=3)
    np.testing.assert_allclose(result, ep.roll_downside_risk(returns, window=3), equal_nan=True)


def test_combined_rolling_regression_aligns_labels_but_other_vectorized_fits_remain_positional():
    dates = pd.bdate_range("2025-01-02", periods=6, tz="UTC")
    benchmark = pd.Series(np.linspace(-0.03, 0.03, 6), index=dates)
    returns = 0.0003 + 1.2 * benchmark
    combined = ep.roll_alpha_beta(returns, benchmark.iloc[::-1], window=3, annualization=12)
    np.testing.assert_allclose(
        combined.to_numpy(), np.tile([1.0003**12 - 1, 1.2], (4, 1)), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        ep.roll_beta(returns, benchmark.iloc[::-1], window=3), [-1.2] * 4, atol=1e-12
    )


def test_combined_rolling_output_cannot_overwrite_either_column_of_original_return_panel():
    panel = np.array(
        [[0.01, 0.02], [-0.01, -0.02], [0.02, 0.03], [0.03, 0.04], [-0.02, -0.03], [0.01, 0.02]]
    )
    original = panel.copy()
    with pytest.raises(ValueError, match="overlap"):
        ep.roll_alpha_beta(panel[:, 0], panel[:, 1], window=3, out=panel[:4])
    np.testing.assert_array_equal(panel, original)


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("annual_volatility", {"alpha": 1.5, "annualization": 52.5}),
        ("sharpe_ratio", {"risk_free": 0.001, "annualization": 12}),
        ("sortino_ratio", {"required_return": 0.001, "_downside_risk": 0.2, "annualization": 12}),
        ("downside_risk", {"required_return": 0.001, "annualization": 12}),
        ("alpha", {"risk_free": 0.001, "_beta": np.full(4, 1.2), "annualization": 12}),
        ("alpha_beta_aligned", {"risk_free": 0.001, "annualization": 12}),
        ("up_capture", {"period": "monthly"}),
    ],
)
def test_published_material_rolling_options_reach_the_shared_kernel(name, kwargs):
    reference = pytest.importorskip("empyrical.stats")
    returns, benchmark = sample()
    args = (returns, benchmark) if name in BINARY else (returns,)
    actual = getattr(ep, "roll_" + name)(*args, window=3, **kwargs)
    expected = getattr(reference, "roll_" + name)(*args, window=3, **kwargs)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11, equal_nan=True)
