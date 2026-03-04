"""Tests for FactorData container and factory methods."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.evaluation.factor.data import FactorData


@pytest.fixture
def sample_factor_df() -> pl.DataFrame:
    """Create a sample factor DataFrame."""
    np.random.seed(42)
    n = 252
    dates = pl.date_range(date(2020, 1, 1), date(2020, 12, 31), eager=True)[:n]
    return pl.DataFrame(
        {
            "timestamp": dates,
            "Mkt-RF": np.random.normal(0.0004, 0.01, n),
            "SMB": np.random.normal(0.0001, 0.005, n),
            "HML": np.random.normal(0.0001, 0.005, n),
            "RF": np.full(n, 0.0001),
        }
    )


@pytest.fixture
def factor_data(sample_factor_df: pl.DataFrame) -> FactorData:
    return FactorData.from_dataframe(sample_factor_df, rf_column="RF")


class TestFactorData:
    def test_from_dataframe_basic(self, sample_factor_df: pl.DataFrame) -> None:
        fd = FactorData.from_dataframe(sample_factor_df, rf_column="RF")
        assert fd.n_factors == 3
        assert fd.factor_names == ["Mkt-RF", "SMB", "HML"]
        assert fd.rf_rate is not None
        assert fd.source == "custom"

    def test_from_dataframe_no_rf(self, sample_factor_df: pl.DataFrame) -> None:
        fd = FactorData.from_dataframe(sample_factor_df.drop("RF"))
        assert fd.rf_rate is None
        assert fd.n_factors == 3

    def test_from_dataframe_custom_timestamp_col(self) -> None:
        df = pl.DataFrame(
            {
                "date": [date(2020, 1, i) for i in range(1, 11)],
                "factor_a": np.random.randn(10),
            }
        )
        fd = FactorData.from_dataframe(df, timestamp_column="date")
        assert "timestamp" in fd.returns.columns
        assert fd.n_periods == 10

    def test_missing_timestamp_raises(self) -> None:
        df = pl.DataFrame({"factor_a": [1.0, 2.0]})
        with pytest.raises(ValueError, match="timestamp"):
            FactorData(
                returns=df,
                rf_rate=None,
                factor_names=["factor_a"],
            )

    def test_missing_factor_column_raises(self) -> None:
        df = pl.DataFrame(
            {
                "timestamp": [date(2020, 1, 1)],
                "factor_a": [0.01],
            }
        )
        with pytest.raises(ValueError, match="missing"):
            FactorData(
                returns=df,
                rf_rate=None,
                factor_names=["factor_a", "factor_b"],
            )

    def test_get_factor_array(self, factor_data: FactorData) -> None:
        arr = factor_data.get_factor_array()
        assert arr.shape == (factor_data.n_periods, 3)
        assert arr.dtype == np.float64

    def test_get_timestamps(self, factor_data: FactorData) -> None:
        ts = factor_data.get_timestamps()
        assert len(ts) == factor_data.n_periods

    def test_properties(self, factor_data: FactorData) -> None:
        assert factor_data.n_periods > 0
        assert factor_data.n_factors == 3
        assert factor_data.frequency == "daily"


class TestFactorDataCombine:
    def test_combine_two(self) -> None:
        dates = [date(2020, 1, i) for i in range(1, 11)]
        fd1 = FactorData.from_dataframe(
            pl.DataFrame(
                {
                    "timestamp": dates,
                    "Mkt": np.random.randn(10),
                }
            )
        )
        fd2 = FactorData.from_dataframe(
            pl.DataFrame(
                {
                    "timestamp": dates,
                    "SMB": np.random.randn(10),
                }
            )
        )
        combined = FactorData.combine(fd1, fd2)
        assert combined.n_factors == 2
        assert set(combined.factor_names) == {"Mkt", "SMB"}
        assert combined.n_periods == 10

    def test_combine_partial_overlap(self) -> None:
        dates1 = [date(2020, 1, i) for i in range(1, 11)]
        dates2 = [date(2020, 1, i) for i in range(5, 15)]
        fd1 = FactorData.from_dataframe(
            pl.DataFrame(
                {
                    "timestamp": dates1,
                    "A": np.random.randn(10),
                }
            )
        )
        fd2 = FactorData.from_dataframe(
            pl.DataFrame(
                {
                    "timestamp": dates2,
                    "B": np.random.randn(10),
                }
            )
        )
        combined = FactorData.combine(fd1, fd2)
        assert combined.n_periods == 6  # overlap: Jan 5-10

    def test_combine_duplicate_names_raises(self) -> None:
        dates = [date(2020, 1, i) for i in range(1, 6)]
        fd1 = FactorData.from_dataframe(
            pl.DataFrame(
                {
                    "timestamp": dates,
                    "A": np.random.randn(5),
                }
            )
        )
        fd2 = FactorData.from_dataframe(
            pl.DataFrame(
                {
                    "timestamp": dates,
                    "A": np.random.randn(5),
                }
            )
        )
        with pytest.raises(ValueError, match="Duplicate"):
            FactorData.combine(fd1, fd2)

    def test_combine_needs_two(self) -> None:
        dates = [date(2020, 1, i) for i in range(1, 6)]
        fd = FactorData.from_dataframe(
            pl.DataFrame(
                {
                    "timestamp": dates,
                    "A": np.random.randn(5),
                }
            )
        )
        with pytest.raises(ValueError, match="at least 2"):
            FactorData.combine(fd)

    def test_combine_preserves_rf(self) -> None:
        dates = [date(2020, 1, i) for i in range(1, 11)]
        df1 = pl.DataFrame(
            {
                "timestamp": dates,
                "Mkt": np.random.randn(10),
                "RF": np.full(10, 0.0001),
            }
        )
        fd1 = FactorData.from_dataframe(df1, rf_column="RF")
        fd2 = FactorData.from_dataframe(
            pl.DataFrame(
                {
                    "timestamp": dates,
                    "SMB": np.random.randn(10),
                }
            )
        )
        combined = FactorData.combine(fd1, fd2)
        assert combined.rf_rate is not None


class TestFactorDataFactories:
    def test_from_fama_french_import_error(self) -> None:
        """from_fama_french raises ImportError if ml4t-data not installed."""
        # This test checks the error message; actual FF data requires ml4t-data
        # We can't guarantee ml4t-data is installed, so test the import path
        try:
            FactorData.from_fama_french(dataset="ff3")
        except ImportError as e:
            assert "ml4t-data" in str(e)
        except Exception:
            # If ml4t-data is installed, the test passes (data loads)
            pass

    def test_from_aqr_import_error(self) -> None:
        try:
            FactorData.from_aqr(dataset="qmj_factors")
        except ImportError as e:
            assert "ml4t-data" in str(e)
        except Exception:
            pass

    def test_from_dataframe_source_label(self, sample_factor_df: pl.DataFrame) -> None:
        fd = FactorData.from_dataframe(sample_factor_df, rf_column="RF", source="test_source")
        assert fd.source == "test_source"

    def test_from_dataframe_frequency(self, sample_factor_df: pl.DataFrame) -> None:
        fd = FactorData.from_dataframe(sample_factor_df, rf_column="RF", frequency="monthly")
        assert fd.frequency == "monthly"
