"""Tests for return attribution and maximal attribution."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.evaluation.factor.attribution import (
    compute_maximal_attribution,
    compute_return_attribution,
)
from ml4t.diagnostic.evaluation.factor.data import FactorData
from ml4t.diagnostic.evaluation.factor.results import (
    AttributionResult,
    MaximalAttributionResult,
)
from ml4t.diagnostic.evaluation.factor.static_model import compute_factor_model


@pytest.fixture
def synthetic_data(
    synthetic_3f_data: tuple[np.ndarray, FactorData],
) -> tuple[np.ndarray, FactorData]:
    """Alias for shared 3-factor fixture."""
    return synthetic_3f_data


class TestReturnAttribution:
    def test_basic(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_return_attribution(returns, fd, window=63, lag=1)

        assert isinstance(result, AttributionResult)
        assert result.window == 63
        assert result.lag == 1
        assert len(result.timestamps) > 0

    def test_contributions_sum(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        """Per-period: factor contribs + alpha + residual = actual return.

        Recover per-period returns from cumulative_total and verify the
        additive identity holds exactly (residual defined as the gap).
        """
        returns, fd = synthetic_data
        result = compute_return_attribution(returns, fd, window=63, lag=1)

        # Recover per-period actual returns from cumulative compounded returns
        cum = result.cumulative_total
        period_returns = np.empty(len(cum))
        period_returns[0] = cum[0]  # (1+r0)-1 = r0
        for t in range(1, len(cum)):
            period_returns[t] = (1 + cum[t]) / (1 + cum[t - 1]) - 1

        n = len(result.timestamps)
        for t in range(n):
            factor_sum = sum(result.factor_contributions[f][t] for f in fd.factor_names)
            reconstructed = factor_sum + result.alpha_contribution[t] + result.residual[t]
            # Identity should hold exactly by construction
            assert abs(reconstructed - period_returns[t]) < 1e-12, (
                f"t={t}: reconstructed={reconstructed:.6e}, actual={period_returns[t]:.6e}"
            )

    def test_summary_pct(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_return_attribution(returns, fd, window=63)

        # Summary percentages should sum to ~100%
        total_pct = sum(result.summary_pct.values())
        assert abs(total_pct - 1.0) < 0.3  # Approximate due to compounding

    def test_attribution_ci(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_return_attribution(returns, fd, window=63, confidence_level=0.95)

        for f in fd.factor_names:
            assert f in result.attribution_se
            assert result.attribution_se[f] > 0
            ci = result.attribution_ci[f]
            assert ci[0] < ci[1]

        assert result.idiosyncratic_se > 0

    def test_lag_matters(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result1 = compute_return_attribution(returns, fd, window=63, lag=1)
        result5 = compute_return_attribution(returns, fd, window=63, lag=5)

        # Different lags should produce different results
        assert len(result1.timestamps) >= len(result5.timestamps)

    def test_to_dataframe(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_return_attribution(returns, fd, window=63)
        df = result.to_dataframe()

        assert isinstance(df, pl.DataFrame)
        assert "timestamp" in df.columns
        for f in fd.factor_names:
            assert f"contrib_{f}" in df.columns

    def test_to_dict(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_return_attribution(returns, fd, window=63)
        d = result.to_dict()

        assert "summary_pct" in d
        assert "attribution_se" in d
        assert "attribution_ci" in d

    def test_summary(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_return_attribution(returns, fd, window=63)
        s = result.summary()

        assert "Return Attribution" in s
        assert "Mkt-RF" in s
        assert "Alpha" in s


class TestMaximalAttribution:
    def test_basic(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_maximal_attribution(returns, fd, factors_of_interest=["Mkt-RF"])

        assert isinstance(result, MaximalAttributionResult)
        assert "Mkt-RF" in result.adjusted_betas
        assert "Mkt-RF" in result.maximal_pnl

    def test_multiple_factors_of_interest(
        self, synthetic_data: tuple[np.ndarray, FactorData]
    ) -> None:
        returns, fd = synthetic_data
        result = compute_maximal_attribution(returns, fd, factors_of_interest=["Mkt-RF", "SMB"])

        assert len(result.adjusted_betas) == 2
        assert len(result.maximal_pnl) == 2

    def test_invalid_factor_raises(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        with pytest.raises(ValueError, match="Unknown"):
            compute_maximal_attribution(returns, fd, factors_of_interest=["INVALID"])

    def test_all_factors_raises(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        with pytest.raises(ValueError, match="outside"):
            compute_maximal_attribution(returns, fd, factors_of_interest=fd.factor_names)

    def test_with_precomputed_model(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        model = compute_factor_model(returns, fd)
        result = compute_maximal_attribution(
            returns, fd, factors_of_interest=["Mkt-RF"], model_result=model
        )
        assert result.adjusted_betas["Mkt-RF"] != 0

    def test_rotation_matrix_shape(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_maximal_attribution(returns, fd, factors_of_interest=["Mkt-RF"])
        # A = Omega_{U,S} @ inv(Omega_{S,S})
        # U has 2 factors (SMB, HML), S has 1 (Mkt-RF)
        assert result.rotation_matrix.shape == (2, 1)

    def test_summary(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_maximal_attribution(returns, fd, factors_of_interest=["Mkt-RF"])
        s = result.summary()
        assert "Maximal Attribution" in s
        assert "Mkt-RF" in s
