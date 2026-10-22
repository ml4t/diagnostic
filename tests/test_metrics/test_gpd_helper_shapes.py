"""Public price-frame workflows must not pool independently owned tail samples."""

import math

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.compat import empyrical as ep


def loss_frame():
    prices = pd.DataFrame(
        {"strategy_a": [100.0, 90.0, 72.0, 36.0], "strategy_b": [200.0, 190.0, 161.5, 80.75]},
        index=pd.bdate_range("2025-01-02", periods=4, tz="UTC"),
    )
    return -ep.simple_returns(prices)


HELPERS = [
    pytest.param(lambda data: ep.gpd_loglikelihood_minimizer_aligned(data), id="fit"),
    pytest.param(lambda data: ep.gpd_loglikelihood([0.1, 1.0], data), id="negative-likelihood"),
    pytest.param(
        lambda data: ep.gpd_loglikelihood_scale_and_shape(0.1, 1.0, data), id="likelihood"
    ),
    pytest.param(lambda data: ep.gpd_loglikelihood_scale_only(0.1, data), id="exponential"),
    pytest.param(
        lambda data: ep.gpd_loglikelihood_factory(data)([0.1, 1.0]), id="negative-factory"
    ),
    pytest.param(
        lambda data: ep.gpd_loglikelihood_scale_and_shape_factory(data)([0.1, 1.0]),
        id="shape-factory",
    ),
    pytest.param(
        lambda data: ep.gpd_loglikelihood_scale_only_factory(data)(0.1), id="scale-factory"
    ),
]


@pytest.mark.parametrize("helper", HELPERS)
def test_two_price_produced_portfolios_cannot_be_pooled_as_one_tail_sample(helper):
    losses = loss_frame()
    original = losses.copy(deep=True)
    with pytest.raises(ValueError, match="one.*column|per column"):
        helper(losses)
    pd.testing.assert_frame_equal(losses, original)


def test_single_column_helpers_keep_one_portfolio_shape_and_independent_density():
    losses = loss_frame()[["strategy_a"]]
    expected = -3 * math.log(0.1) - 2 * sum(math.log(1 + value / 0.1) for value in [0.1, 0.2, 0.5])
    assert ep.gpd_loglikelihood_scale_and_shape(0.1, 1, losses) == pytest.approx(expected)
    assert ep.gpd_loglikelihood_factory(losses)([0.1, 1]) == pytest.approx(-expected)
    values = losses.to_numpy()
    np.testing.assert_allclose(
        ep.gpd_loglikelihood_minimizer_aligned(losses),
        ep.gpd_loglikelihood_minimizer_aligned(values[:, 0]),
    )


def test_separate_price_produced_columns_keep_reference_fit_and_likelihood_contracts():
    reference = pytest.importorskip("empyrical.stats")
    for _, values in loss_frame().items():
        np.testing.assert_allclose(
            ep.gpd_loglikelihood_minimizer_aligned(values),
            reference.gpd_loglikelihood_minimizer_aligned(values),
            rtol=1e-5,
            atol=1e-10,
        )
        assert ep.gpd_loglikelihood_scale_and_shape(0.1, 1, values) == pytest.approx(
            reference.gpd_loglikelihood_scale_and_shape(0.1, 1, values)
        )
