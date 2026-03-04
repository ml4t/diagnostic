"""Shared fixtures for evaluation tests.

Factor module fixtures are shared across test_factor_*.py files.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.evaluation.factor.data import FactorData


@pytest.fixture
def synthetic_3f_data() -> tuple[np.ndarray, FactorData]:
    """Synthetic 3-factor data with known betas.

    True model: r_t = 0.0002 + 1.0*Mkt-RF + 0.3*SMB - 0.1*HML + eps_t
    eps ~ N(0, 0.003), T=500, seed=42
    """
    np.random.seed(42)
    T = 500

    mkt = np.random.normal(0.0004, 0.01, T)
    smb = np.random.normal(0.0001, 0.005, T)
    hml = np.random.normal(0.0001, 0.005, T)
    eps = np.random.normal(0, 0.003, T)
    returns = 0.0002 + 1.0 * mkt + 0.3 * smb - 0.1 * hml + eps

    dates = pl.date_range(date(2018, 1, 1), date(2019, 12, 31), eager=True)[:T]
    factor_df = pl.DataFrame(
        {"timestamp": dates, "Mkt-RF": mkt, "SMB": smb, "HML": hml}
    )
    return returns, FactorData.from_dataframe(factor_df)


@pytest.fixture
def synthetic_2f_data() -> tuple[np.ndarray, FactorData]:
    """Synthetic 2-factor data with known betas.

    True model: r_t = 0.0002 + 1.0*Mkt-RF + 0.3*SMB + eps_t
    eps ~ N(0, 0.003), T=500, seed=42
    """
    np.random.seed(42)
    T = 500

    mkt = np.random.normal(0.0004, 0.01, T)
    smb = np.random.normal(0.0001, 0.005, T)
    eps = np.random.normal(0, 0.003, T)
    returns = 0.0002 + 1.0 * mkt + 0.3 * smb + eps

    dates = pl.date_range(date(2018, 1, 1), date(2019, 12, 31), eager=True)[:T]
    factor_df = pl.DataFrame({"timestamp": dates, "Mkt-RF": mkt, "SMB": smb})
    return returns, FactorData.from_dataframe(factor_df)
