"""Tests for factor timing analysis (Tier 2)."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.evaluation.factor.data import FactorData
from ml4t.diagnostic.evaluation.factor.results import FactorTimingResult
from ml4t.diagnostic.evaluation.factor.timing import compute_factor_timing


@pytest.fixture
def synthetic_data() -> tuple[np.ndarray, FactorData]:
    np.random.seed(42)
    T = 500
    mkt = np.random.normal(0.0004, 0.01, T)
    smb = np.random.normal(0.0001, 0.005, T)
    eps = np.random.normal(0, 0.003, T)
    returns = 0.0002 + 1.0 * mkt + 0.3 * smb + eps

    dates = pl.date_range(date(2018, 1, 1), date(2019, 12, 31), eager=True)[:T]
    factor_df = pl.DataFrame(
        {
            "timestamp": dates,
            "Mkt-RF": mkt,
            "SMB": smb,
        }
    )
    return returns, FactorData.from_dataframe(factor_df)


class TestFactorTiming:
    def test_basic(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_factor_timing(returns, fd, window=63)

        assert isinstance(result, FactorTimingResult)
        assert result.window == 63
        assert len(result.factor_names) == 2

    def test_correlations_in_range(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_factor_timing(returns, fd, window=63)

        for f in fd.factor_names:
            corr = result.correlations[f]
            if np.isfinite(corr):
                assert -1 <= corr <= 1

    def test_p_values(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_factor_timing(returns, fd, window=63)

        for f in fd.factor_names:
            p = result.p_values[f]
            if np.isfinite(p):
                assert 0 <= p <= 1

    def test_to_dataframe(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_factor_timing(returns, fd, window=63)
        df = result.to_dataframe()

        assert isinstance(df, pl.DataFrame)
        assert "factor" in df.columns
        assert "timing_correlation" in df.columns

    def test_summary(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_factor_timing(returns, fd, window=63)
        s = result.summary()
        assert "Factor Timing" in s
