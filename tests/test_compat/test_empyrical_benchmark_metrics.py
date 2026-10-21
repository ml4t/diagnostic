"""A dated price-to-return user workflow exercises actual benchmark-relative APIs."""

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.compat import empyrical as ep


def price_workflow():
    dates = pd.date_range("2025-01-02", periods=6, freq="B", tz="UTC")
    benchmark = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
    strategy = 0.0003 + 1.2 * benchmark
    prices = np.concatenate(([100.0], 100 * np.cumprod(1 + strategy)))
    returns = pd.Series(ep.simple_returns(prices), index=dates[1:], name="strategy")
    factor = pd.Series(benchmark, index=dates[1:], name="external_benchmark")
    return returns, factor


def test_dated_price_workflow_keeps_compounded_alpha_and_unannualized_active_sharpe():
    returns, benchmark = price_workflow()
    original = returns.copy(deep=True)
    result = ep.alpha_beta(returns, benchmark.iloc[::-1], annualization=252)
    assert result == pytest.approx([(1.0003) ** 252 - 1, 1.2], abs=1e-12, rel=1e-12)
    active = returns.to_numpy() - benchmark.to_numpy()
    assert ep.excess_sharpe(returns, benchmark) == pytest.approx(active.mean() / active.std(ddof=1))
    pd.testing.assert_series_equal(returns, original)


def test_public_up_down_metrics_use_strict_benchmark_signs_not_strategy_signs():
    returns, benchmark = price_workflow()
    expected = [(1.0003) ** 12 - 1, 1.2]
    assert ep.up_alpha_beta(returns, benchmark, annualization=12) == pytest.approx(
        expected, abs=1e-12
    )
    assert ep.down_alpha_beta(returns, benchmark, annualization=12) == pytest.approx(
        expected, abs=1e-12
    )


def test_native_arithmetic_alpha_and_annual_information_ratio_stay_distinct():
    from ml4t.diagnostic.evaluation.portfolio_analysis import alpha_beta, information_ratio

    returns, benchmark = price_workflow()
    assert alpha_beta(returns, benchmark) == pytest.approx([0.0756, 1.2], abs=1e-12)
    assert information_ratio(returns, benchmark) == pytest.approx(
        ep.excess_sharpe(returns, benchmark) * np.sqrt(252)
    )
    positional = ep.alpha_beta_aligned(returns, benchmark.iloc[::-1])
    assert positional == pytest.approx([(1.0003) ** 252 - 1, -1.2], abs=1e-12)


def test_periodic_risk_free_missingness_does_not_refit_raw_benchmark_beta():
    returns, benchmark = price_workflow()
    risk_free = np.array([0.0001, 0.0, 0.0002, np.nan, 0.0001])
    actual = ep.alpha_beta(returns.to_numpy(), benchmark.to_numpy(), risk_free, annualization=12)
    assert actual == pytest.approx([(1.00032) ** 12 - 1, 1.2], abs=1e-12)


def test_missing_pairs_and_supplied_slope_have_different_valid_sample_guards():
    benchmark = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
    returns = 0.0003 + 1.2 * benchmark
    returns[1] = np.nan
    benchmark[3] = np.nan
    assert ep.alpha_beta_aligned(returns, benchmark) == pytest.approx(
        [(1.0003) ** 252 - 1, 1.2], abs=1e-12
    )
    returns[2:] = np.nan
    assert np.isnan(ep.beta_aligned(returns, benchmark))
    assert ep.alpha_aligned(returns, benchmark, _beta=1.2) == pytest.approx((1.0003) ** 252 - 1)


def test_legacy_variance_floor_does_not_change_native_nonconstant_benchmark_fit():
    from ml4t.diagnostic.evaluation.portfolio_analysis import alpha_beta

    benchmark = np.linspace(-1, 1, 5) * 1e-16
    returns = 1.2 * benchmark
    assert np.isnan(ep.alpha_beta_aligned(returns, benchmark)).all()
    assert alpha_beta(returns, benchmark)[1] == pytest.approx(1.2)


def test_aligned_matrix_columns_and_labeled_alpha_are_not_pooled():
    benchmark = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
    returns = np.column_stack([0.0003 + 1.2 * benchmark, 0.0002 - 0.4 * benchmark])
    expected = np.array([[(1.0003) ** 252 - 1, 1.2], [(1.0002) ** 252 - 1, -0.4]])
    assert ep.alpha_beta_aligned(returns, benchmark[:, None]) == pytest.approx(expected, abs=1e-12)
    frame = pd.DataFrame(returns, columns=["long", "hedge"])
    alpha = ep.alpha_aligned(frame, benchmark[:, None])
    assert isinstance(alpha, pd.Series) and list(alpha.index) == ["long", "hedge"]
    assert alpha.to_numpy() == pytest.approx(expected[:, 0])
    assert isinstance(ep.beta_aligned(frame, benchmark), np.ndarray)


def test_one_column_frame_unaligned_output_shape_follows_the_actual_series_route():
    returns, benchmark = price_workflow()
    output = np.full(2, -99.0)
    actual = ep.alpha_beta(returns.to_frame(), benchmark, out=output)
    assert actual is output
    assert actual == pytest.approx([(1.0003) ** 252 - 1, 1.2], abs=1e-12)


@pytest.mark.parametrize(
    "name", ["alpha_beta", "alpha_beta_aligned", "up_alpha_beta", "down_alpha_beta"]
)
def test_original_benchmark_buffer_alias_is_rejected_before_any_selected_copy(name):
    returns, benchmark = (value.to_numpy(copy=True) for value in price_workflow())
    original = benchmark.copy()
    with pytest.raises(ValueError, match="overlap"):
        getattr(ep, name)(returns, benchmark, out=benchmark[:2])
    np.testing.assert_array_equal(benchmark, original)


def test_scalar_output_alias_to_original_float32_series_is_not_hidden_by_normalization():
    returns, benchmark = price_workflow()
    storage = benchmark.to_numpy(dtype=np.float32, copy=True)
    benchmark = pd.Series(storage, index=benchmark.index, copy=False)
    original = benchmark.copy(deep=True)
    assert np.shares_memory(storage, benchmark.to_numpy(copy=False))
    with pytest.raises(ValueError, match="overlap"):
        ep.alpha(returns, benchmark, out=storage[:1].reshape(()))
    pd.testing.assert_series_equal(benchmark, original)


def test_invalid_aligned_input_does_not_partially_fill_output():
    output = np.array([123.0, 456.0])
    with pytest.raises(ValueError, match="row counts"):
        ep.alpha_beta_aligned(np.ones(5), np.ones(4), out=output)
    np.testing.assert_array_equal(output, [123.0, 456.0])
    with pytest.raises(ValueError, match="annualization"):
        ep.alpha_beta_aligned(np.ones(5), np.ones(5), annualization=-1, out=output)
    np.testing.assert_array_equal(output, [123.0, 456.0])


@pytest.mark.parametrize("constant,expected", [(1.0, np.inf), (-1.0, -np.inf), (0.0, np.nan)])
def test_constant_active_return_excess_sharpe_keeps_its_ratio_domain(constant, expected):
    actual = ep.excess_sharpe(np.full(5, constant), np.zeros(5))
    if np.isnan(expected):
        assert np.isnan(actual)
    else:
        assert actual == expected


def test_batting_average_aligns_dated_pairs_and_exposes_neutral_market_convention():
    from ml4t.diagnostic.evaluation.portfolio_analysis import benchmark_batting_average

    dates = pd.date_range("2025-01-02", periods=5, tz="UTC")
    returns = pd.Series([0.02, 0.0, -0.01, 0.01], index=dates[:4])
    benchmark = pd.Series([0.01, 0.0, -0.02, 0.03], index=dates[[0, 1, 2, 4]])
    result = benchmark_batting_average(returns, benchmark)
    assert result["n_observations"] == 5 and result["n_paired"] == 3 and result["n_excluded"] == 2
    assert result["overall"] == 2 / 3 and result["up"] == 1 and result["down"] == 1
    assert result["n_zero_benchmark"] == 1 and result["n_up"] == 1
    legacy = ep.batting_average(returns, benchmark)
    assert list(legacy.index) == ["batting average", "up market", "down market"]
    assert legacy.to_numpy() == pytest.approx([2 / 3, 0.5, 1])


def test_batting_counts_omit_nonfinite_active_pairs_and_do_not_count_ties_as_wins():
    from ml4t.diagnostic.evaluation.portfolio_analysis import benchmark_batting_average

    returns = np.array([0.02, 0.0, -0.01, 0.03, 0.05, np.inf])
    benchmark = np.array([0.01, 0.0, -0.02, 0.03, np.nan, 0.04])
    result = benchmark_batting_average(returns, benchmark)
    assert result["n_paired"] == 4 and result["n_overall_wins"] == 2
    assert result["overall"] == 0.5 and result["up"] == 0.5 and result["down"] == 1
    assert ep.batting_average(returns, benchmark).to_numpy() == pytest.approx([0.5, 1 / 3, 1])


@pytest.mark.parametrize("shape", [(0,), (1,), (0, 2)])
def test_empty_and_insufficient_regression_preserve_per_function_output_shapes(shape):
    returns = np.zeros(shape)
    benchmark = np.zeros(shape)
    result = ep.alpha_beta_aligned(returns, benchmark)
    assert result.shape == shape[1:] + (2,)
    assert np.isnan(result).all()


@pytest.mark.parametrize(
    "name",
    [
        "alpha_beta",
        "alpha_beta_aligned",
        "alpha",
        "alpha_aligned",
        "beta",
        "beta_aligned",
        "excess_sharpe",
        "up_alpha_beta",
        "down_alpha_beta",
        "batting_average",
    ],
)
def test_actual_published_benchmark_workflow_on_shared_dated_inputs(name):
    reference = pytest.importorskip("empyrical")
    returns, benchmark = price_workflow()
    kwargs = (
        {"annualization": 52.5}
        if name not in ("beta", "beta_aligned", "excess_sharpe", "batting_average")
        else {}
    )
    actual = getattr(ep, name)(returns, benchmark, **kwargs)
    expected = getattr(reference, name)(returns.copy(), benchmark.copy(), **kwargs)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("name", ["alpha_beta_aligned", "alpha_aligned", "beta_aligned"])
def test_actual_published_matrix_contract_uses_explicit_benchmark_column(name):
    reference = pytest.importorskip("empyrical")
    _, source = price_workflow()
    benchmark = source.to_numpy()[:, None]
    returns = np.column_stack([0.0003 + 1.2 * benchmark[:, 0], 0.0002 - 0.4 * benchmark[:, 0]])
    returns[1, 0] = np.nan
    actual = getattr(ep, name)(returns, benchmark)
    expected = getattr(reference, name)(returns.copy(), benchmark.copy())
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
