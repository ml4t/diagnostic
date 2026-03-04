"""Performance benchmarks for the factor module.

Uses pytest-benchmark to track performance of key operations.
Run with: uv run pytest tests/test_performance/test_factor_benchmarks.py -v --benchmark-only
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.evaluation.factor.data import FactorData
from ml4t.diagnostic.evaluation.factor.kalman import compute_kalman_betas
from ml4t.diagnostic.evaluation.factor.rolling_model import compute_rolling_exposures
from ml4t.diagnostic.evaluation.factor.static_model import compute_factor_model


def _make_data(n_obs: int, n_factors: int) -> tuple[np.ndarray, FactorData]:
    """Generate synthetic factor data of given size."""
    np.random.seed(42)
    factors = np.random.normal(0, 0.01, (n_obs, n_factors))
    betas = np.random.uniform(0.2, 1.5, n_factors)
    eps = np.random.normal(0, 0.003, n_obs)
    returns = 0.0002 + factors @ betas + eps

    dates = pl.date_range(date(2000, 1, 1), date(2050, 12, 31), eager=True)[:n_obs]
    cols: dict = {"timestamp": dates}
    for k in range(n_factors):
        cols[f"F{k}"] = factors[:, k]
    fd = FactorData.from_dataframe(pl.DataFrame(cols))
    return returns, fd


@pytest.fixture
def data_small() -> tuple[np.ndarray, FactorData]:
    """500 obs, 3 factors (typical backtest)."""
    return _make_data(500, 3)


@pytest.fixture
def data_medium() -> tuple[np.ndarray, FactorData]:
    """2500 obs, 5 factors (~10 years daily with 5 factors)."""
    return _make_data(2500, 5)


@pytest.fixture
def data_large() -> tuple[np.ndarray, FactorData]:
    """5000 obs, 3 factors (~20 years daily)."""
    return _make_data(5000, 3)


class TestStaticModelBenchmarks:
    @pytest.mark.benchmark(group="static_ols")
    def test_static_ols_small(self, benchmark, data_small: tuple) -> None:
        returns, fd = data_small
        result = benchmark(compute_factor_model, returns, fd, hac=False)
        assert result.r_squared > 0

    @pytest.mark.benchmark(group="static_ols")
    def test_static_ols_medium(self, benchmark, data_medium: tuple) -> None:
        returns, fd = data_medium
        result = benchmark(compute_factor_model, returns, fd, hac=False)
        assert result.r_squared > 0

    @pytest.mark.benchmark(group="static_ols")
    def test_static_hac_medium(self, benchmark, data_medium: tuple) -> None:
        returns, fd = data_medium
        result = benchmark(compute_factor_model, returns, fd, hac=True)
        assert result.hac is True


class TestRollingModelBenchmarks:
    @pytest.mark.benchmark(group="rolling_ols")
    def test_rolling_small(self, benchmark, data_small: tuple) -> None:
        returns, fd = data_small
        result = benchmark(compute_rolling_exposures, returns, fd, window=63)
        assert len(result.timestamps) > 0

    @pytest.mark.benchmark(group="rolling_ols")
    def test_rolling_medium(self, benchmark, data_medium: tuple) -> None:
        returns, fd = data_medium
        result = benchmark(compute_rolling_exposures, returns, fd, window=63)
        assert len(result.timestamps) > 0

    @pytest.mark.benchmark(group="rolling_ols")
    def test_rolling_large(self, benchmark, data_large: tuple) -> None:
        returns, fd = data_large
        result = benchmark(compute_rolling_exposures, returns, fd, window=63)
        assert len(result.timestamps) > 0

    @pytest.mark.benchmark(group="rolling_ols")
    def test_rolling_large_wide_window(self, benchmark, data_large: tuple) -> None:
        returns, fd = data_large
        result = benchmark(compute_rolling_exposures, returns, fd, window=252)
        assert len(result.timestamps) > 0


class TestKalmanBenchmarks:
    @pytest.mark.benchmark(group="kalman")
    def test_kalman_small_no_optimize(self, benchmark, data_small: tuple) -> None:
        returns, fd = data_small
        result = benchmark(compute_kalman_betas, returns, fd, optimize_noise=False)
        assert result.window == 0

    @pytest.mark.benchmark(group="kalman")
    def test_kalman_medium_no_optimize(self, benchmark, data_medium: tuple) -> None:
        returns, fd = data_medium
        result = benchmark(compute_kalman_betas, returns, fd, optimize_noise=False)
        assert result.window == 0
