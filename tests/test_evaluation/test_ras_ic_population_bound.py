"""Prevent false IC confidence from mean-scale or sample-selected bounds."""

import warnings
from datetime import date
from math import comb

import numpy as np
import polars as pl
import pytest
from scipy.stats import binom

from ml4t.diagnostic.config import RASSettings, SignalConfig
from ml4t.diagnostic.evaluation.stats import RASResult, ras_ic_adjustment
from ml4t.diagnostic.metrics import cross_sectional_ic_series
from ml4t.diagnostic.visualization.backtest import plot_ras_analysis


def test_default_covers_bounded_correlations_in_exact_binomial_experiment():
    """Mean-scale default missed truth in 22.8% of this genuine IC population."""
    # Orthogonal centered cross sections realize correlations -0.1 and +0.1.
    u = np.array([-1.0, -1.0, 1.0, 1.0])
    v = np.array([-1.0, 1.0, -1.0, 1.0])
    predictions = pl.DataFrame(
        {
            "date": [date(2026, 1, 1)] * 4 + [date(2026, 1, 2)] * 4,
            "asset": list("abcd") * 2,
            "prediction": np.tile(u, 2),
        }
    )
    returns = predictions.select("date", "asset").with_columns(
        pl.Series(
            "forward_return", np.concatenate([r * u + np.sqrt(1 - r * r) * v for r in [-0.1, 0.1]])
        )
    )
    ic = cross_sectional_ic_series(
        predictions, returns, entity_col="asset", method="pearson", min_obs=4
    )["ic"].to_numpy()
    np.testing.assert_allclose(ic, [-0.1, 0.1], atol=1e-15)
    # Enumerate every possible count, weighted by its known population probability.
    n = 252
    counts = np.arange(n + 1)
    means = (counts * ic[1] + (n - counts) * ic[0]) / n
    truth = 0.55 * ic[1] + 0.45 * ic[0]
    probability = binom.pmf(counts, n, 0.55)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        default = np.array([ras_ic_adjustment(np.array([m]), 0.0, n)[0] for m in means])
        understated = np.array(
            [ras_ic_adjustment(np.array([m]), 0.0, n, kappa=0.02)[0] for m in means]
        )
        valid = np.array([ras_ic_adjustment(np.array([m]), 0.0, n, kappa=0.1)[0] for m in means])
    assert probability[default > truth].sum() <= 0.05
    assert probability[valid > truth].sum() <= 0.05
    assert probability[understated > truth].sum() > 0.05


def test_rare_unobserved_ic_invalidates_sample_max_as_population_certificate():
    """A finite sample can miss a valid -1 IC with probability above delta."""
    n = 252
    samples = np.full((n, 1), 0.001)
    population_mean = -0.01 + 0.99 * 0.001
    assert 0.99**n > 0.05  # Exact probability that all observations equal +.001.
    # nextafter avoids a floating-point mean comparison at the sample maximum.
    sample_bound = np.nextafter(samples.max(), np.inf)
    unsafe = ras_ic_adjustment(samples.mean(0), 0.0, n, kappa=sample_bound)
    safe = ras_ic_adjustment(samples.mean(0), 0.0, n)
    assert unsafe[0] > population_mean
    assert safe[0] <= population_mean
    with pytest.raises(ValueError, match="kappa must be positive and finite"):
        ras_ic_adjustment(samples.mean(0), 0.0, n, kappa=None, ic_samples=samples)


def test_simultaneous_coverage_for_two_correlated_strategies_uses_exact_complexity():
    """Selection of either sign of the same IC needs a joint confidence bound."""
    n = 252
    counts = np.arange(n + 1)
    # For columns x and -x with |x[t]|=.1, the maximum signed dot product
    # is .1*|S_n|/n for a symmetric sign walk, irrespective of the sample x.
    exact_complexity = 0.1 * sum(comb(n, k) * abs(2 * k - n) for k in range(n + 1)) / 2**n / n
    means = 0.1 * (2 * counts - n) / n
    truth = np.array([0.01, -0.01])
    misses = [
        np.any(ras_ic_adjustment(np.array([m, -m]), exact_complexity, n, kappa=0.1) > truth)
        for m in means
    ]
    # Include every tail outcome, with known population rather than simulated weights.
    noncoverage = binom.pmf(counts, n, 0.55)[misses].sum()
    assert 0 < noncoverage <= 0.05


def test_observed_period_violation_is_rejected_even_with_zero_mean():
    """Cancelling positive/negative ICs must not hide an understated bound."""
    samples = np.array([[-0.3], [0.3]])
    with pytest.raises(ValueError, match="period ICs.*exceed kappa"):
        ras_ic_adjustment(samples.mean(0), 0.0, 2, kappa=0.02, ic_samples=samples)


def test_valid_population_bound_retains_negative_values_and_positional_contract():
    """Externally justified bounds and old positional result requests still work."""
    samples = np.array([[-0.1, 0.1], [0.1, -0.1]])
    result = ras_ic_adjustment(samples.mean(0), 0.1, 2, 0.05, 0.1, True)
    assert isinstance(result, RASResult)
    assert np.all(result.adjusted_values < 0)
    assert result.n_significant == 0
    # Mohri et al. Theorem 3.3 rescaled from [0,1] to [-.1,.1], T=2, delta=.05.
    assert result.estimation_error == pytest.approx(0.5761936747919524)
    assert result.massart_bound == pytest.approx(0.08325546111576977)
    checked = ras_ic_adjustment(samples.mean(0), 0.1, 2, 0.05, 0.1, ic_samples=samples)
    np.testing.assert_array_equal(checked, result.adjusted_values)


def test_mean_only_warning_does_not_describe_an_empirical_population_certificate():
    """Preserve the old warning path, but identify the actual invalid assumption."""
    with pytest.warns(UserWarning, match="population bound on period ICs is violated"):
        ras_ic_adjustment(np.array([0.3]), 0.0, 252, kappa=0.02)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_nonfinite_or_nonpositive_bound_cannot_yield_a_confidence_result(value):
    with pytest.raises(ValueError, match="kappa must be positive and finite"):
        ras_ic_adjustment(np.array([0.0]), 0.0, 2, kappa=value)


@pytest.mark.parametrize(
    "samples, message",
    [
        (np.zeros((3, 1)), "shape"),
        (np.array([[np.nan], [0.0]]), "finite period ICs"),
        (np.array([[0.1], [0.1]]), "column means"),
    ],
)
def test_diagnostic_period_matrix_cannot_describe_a_different_sample(samples, message):
    with pytest.raises(ValueError, match=message):
        ras_ic_adjustment(np.array([0.0]), 0.0, 2, ic_samples=samples)


@pytest.mark.parametrize("means", [np.array([]), np.array([np.nan]), np.array([np.inf])])
def test_missing_means_cannot_produce_a_silent_confidence_result(means):
    with pytest.raises(ValueError, match="finite, nonempty"):
        ras_ic_adjustment(means, 0.0, 2)


@pytest.mark.parametrize("n", [True, 2.5, np.inf])
def test_period_count_is_an_actual_positive_integer(n):
    with pytest.raises(ValueError, match="positive integer"):
        ras_ic_adjustment(np.array([0.0]), 0.0, n)


@pytest.mark.parametrize("complexity", [np.nan, np.inf, -0.1])
def test_invalid_complexity_cannot_produce_a_confidence_result(complexity):
    with pytest.raises(ValueError, match="finite and non-negative"):
        ras_ic_adjustment(np.array([0.0]), complexity, 2)


def test_configuration_uses_correlation_population_bound_and_preserves_explicit_values():
    default = SignalConfig()
    assert default.ras.kappa == 1.0
    restored = SignalConfig.model_validate_json(default.model_dump_json())
    assert restored.ras.kappa == 1.0
    assert RASSettings(kappa=0.1).kappa == 0.1


def test_public_result_chart_uses_actual_estimation_penalty_instead_of_twice_kappa():
    """The visible waterfall must add to the public API lower bound, including negatives."""
    result = ras_ic_adjustment(np.array([0.05]), 0.01, 252, return_result=True)
    figure = plot_ras_analysis(0.05, result.adjusted_values[0], result.complexity)
    waterfall = figure.data[0]
    assert waterfall.x[2] == "Estimation error"
    assert waterfall.y[2] == pytest.approx(-result.estimation_error)
    assert sum(waterfall.y[:3]) == pytest.approx(result.adjusted_values[0])
    assert waterfall.y[-1] < 0


def test_inconsistent_chart_inputs_do_not_imply_a_valid_confidence_bound():
    with pytest.raises(ValueError, match="non-negative complexity and penalties"):
        plot_ras_analysis(0.05, 0.03, 0.02)
