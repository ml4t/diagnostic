"""FactorData container and factory methods for factor model inputs.

Provides a unified container for factor return data with factory methods
for common sources (Fama-French, AQR) via lazy imports from ml4t-data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class FactorData:
    """Container for factor return data aligned to a common timeline.

    Parameters
    ----------
    returns : pl.DataFrame
        DataFrame with a 'timestamp' column and one column per factor.
        Values must be in decimal format (0.01 = 1%).
    rf_rate : pl.Series | None
        Risk-free rate series aligned to returns timestamps.
        In decimal format (0.0001 = 1bp daily).
    factor_names : list[str]
        Names of factor columns in returns.
    source : str
        Data source identifier.
    frequency : str
        Return frequency: "daily", "weekly", "monthly".
    """

    returns: pl.DataFrame
    rf_rate: pl.Series | None
    factor_names: list[str]
    source: str = "custom"
    frequency: str = "daily"

    def __post_init__(self) -> None:
        if "timestamp" not in self.returns.columns:
            raise ValueError("returns DataFrame must contain a 'timestamp' column")
        missing = [f for f in self.factor_names if f not in self.returns.columns]
        if missing:
            raise ValueError(f"Factor columns missing from returns: {missing}")
        # Ensure timestamp is a sortable/joinable type (Date or Datetime)
        ts_dtype = self.returns["timestamp"].dtype
        if ts_dtype in (pl.Object, pl.Utf8, pl.String):
            try:
                self.returns = self.returns.with_columns(pl.col("timestamp").cast(pl.Date))
            except Exception:
                # If casting to Date fails, try Datetime
                self.returns = self.returns.with_columns(pl.col("timestamp").cast(pl.Datetime))

    @property
    def n_periods(self) -> int:
        return len(self.returns)

    @property
    def n_factors(self) -> int:
        return len(self.factor_names)

    def get_factor_array(self) -> np.ndarray:
        """Factor returns as (T, K) numpy array."""
        return self.returns.select(self.factor_names).to_numpy()

    def get_timestamps(self) -> np.ndarray:
        """Timestamp column as numpy array."""
        return self.returns["timestamp"].to_numpy()

    @classmethod
    def from_dataframe(
        cls,
        df: pl.DataFrame,
        *,
        rf_column: str | None = None,
        timestamp_column: str = "timestamp",
        source: str = "custom",
        frequency: str = "daily",
    ) -> FactorData:
        """Create FactorData from a Polars DataFrame.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame with timestamp and factor return columns.
        rf_column : str | None
            Column name for risk-free rate. Extracted and removed from factors.
        timestamp_column : str
            Column name for timestamps.
        source : str
            Data source label.
        frequency : str
            Return frequency.
        """
        if timestamp_column != "timestamp":
            df = df.rename({timestamp_column: "timestamp"})

        rf_rate = None
        if rf_column and rf_column in df.columns:
            rf_rate = df[rf_column]
            df = df.drop(rf_column)

        factor_names = [c for c in df.columns if c != "timestamp"]
        return cls(
            returns=df,
            rf_rate=rf_rate,
            factor_names=factor_names,
            source=source,
            frequency=frequency,
        )

    @classmethod
    def from_fama_french(
        cls,
        dataset: str = "ff3",
        frequency: str = "daily",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> FactorData:
        """Load Fama-French factor data via ml4t-data.

        Parameters
        ----------
        dataset : str
            Dataset identifier: "ff3" (3-factor), "ff5" (5-factor),
            "momentum" (MOM factor).
        frequency : str
            "daily" or "monthly".
        start_date, end_date : str | None
            Date range filter (YYYY-MM-DD).

        Raises
        ------
        ImportError
            If ml4t-data is not installed.
        """
        try:
            from ml4t.data.providers import FamaFrenchProvider
        except ImportError as e:
            raise ImportError(
                "ml4t-data is required for Fama-French data. Install with: pip install ml4t-data"
            ) from e

        provider = FamaFrenchProvider()

        dataset_map = {
            "ff3": "F-F_Research_Data_Factors",
            "ff5": "F-F_Research_Data_5_Factors_2x3",
            "momentum": "F-F_Momentum_Factor",
        }
        ff_dataset = dataset_map.get(dataset, dataset)
        suffix = "_daily" if frequency == "daily" else ""
        full_name = ff_dataset + suffix

        raw = provider.get_data(full_name)

        # Fama-French data is in percent, convert to decimal
        if isinstance(raw, pl.DataFrame):
            df = raw
        else:
            df = pl.from_pandas(raw)

        # Identify timestamp column
        ts_col = None
        for c in df.columns:
            if c.lower() in ("date", "timestamp", "period"):
                ts_col = c
                break
        if ts_col is None:
            ts_col = df.columns[0]

        df = df.rename({ts_col: "timestamp"})

        # Convert percent to decimal for numeric columns
        numeric_cols = [
            c
            for c in df.columns
            if c != "timestamp" and df[c].dtype in (pl.Float64, pl.Float32, pl.Int64)
        ]
        df = df.with_columns([(pl.col(c) / 100.0).alias(c) for c in numeric_cols])

        # Filter date range
        if start_date:
            df = df.filter(pl.col("timestamp") >= pl.lit(start_date).str.to_date())
        if end_date:
            df = df.filter(pl.col("timestamp") <= pl.lit(end_date).str.to_date())

        rf_column = "RF" if "RF" in df.columns else None
        return cls.from_dataframe(
            df, rf_column=rf_column, source=f"fama_french_{dataset}", frequency=frequency
        )

    @classmethod
    def from_aqr(
        cls,
        dataset: str = "qmj_factors",
        region: str = "USA",
        frequency: str = "daily",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> FactorData:
        """Load AQR factor data via ml4t-data.

        Parameters
        ----------
        dataset : str
            AQR dataset identifier.
        region : str
            Geographic region filter.
        frequency : str
            "daily" or "monthly".
        start_date, end_date : str | None
            Date range filter.

        Raises
        ------
        ImportError
            If ml4t-data is not installed.
        """
        try:
            from ml4t.data.providers import AQRFactorProvider
        except ImportError as e:
            raise ImportError(
                "ml4t-data is required for AQR factor data. Install with: pip install ml4t-data"
            ) from e

        provider = AQRFactorProvider()
        raw = provider.get_data(dataset, region=region, frequency=frequency)

        if isinstance(raw, pl.DataFrame):
            df = raw
        else:
            df = pl.from_pandas(raw)

        ts_col = None
        for c in df.columns:
            if c.lower() in ("date", "timestamp", "period"):
                ts_col = c
                break
        if ts_col is None:
            ts_col = df.columns[0]

        df = df.rename({ts_col: "timestamp"})

        if start_date:
            df = df.filter(pl.col("timestamp") >= pl.lit(start_date).str.to_date())
        if end_date:
            df = df.filter(pl.col("timestamp") <= pl.lit(end_date).str.to_date())

        return cls.from_dataframe(df, source=f"aqr_{dataset}", frequency=frequency)

    @classmethod
    def combine(cls, *factor_data_items: FactorData) -> FactorData:
        """Combine multiple FactorData via inner join on timestamp.

        Parameters
        ----------
        *factor_data_items : FactorData
            Two or more FactorData instances to combine.

        Returns
        -------
        FactorData
            Combined data with all factors, aligned timestamps.
        """
        if len(factor_data_items) < 2:
            raise ValueError("Need at least 2 FactorData items to combine")

        result_df = factor_data_items[0].returns
        all_names = list(factor_data_items[0].factor_names)
        rf_rate = factor_data_items[0].rf_rate
        sources = [factor_data_items[0].source]

        for fd in factor_data_items[1:]:
            # Check for duplicate factor names
            overlap = set(all_names) & set(fd.factor_names)
            if overlap:
                raise ValueError(f"Duplicate factor names: {overlap}")

            result_df = result_df.join(
                fd.returns.select(["timestamp"] + fd.factor_names),
                on="timestamp",
                how="inner",
            )
            all_names.extend(fd.factor_names)
            sources.append(fd.source)

            if rf_rate is None and fd.rf_rate is not None:
                rf_rate = fd.rf_rate

        # Re-align rf_rate to joined timestamps if needed
        if rf_rate is not None and len(rf_rate) != len(result_df):
            # Find the source DataFrame that provided rf_rate
            rf_source = None
            for fd in factor_data_items:
                if fd.rf_rate is not None:
                    rf_source = fd
                    break
            if rf_source is not None:
                rf_df = pl.DataFrame(
                    {
                        "timestamp": rf_source.returns["timestamp"],
                        "_rf": rf_source.rf_rate,
                    }
                )
                joined = result_df.select("timestamp").join(rf_df, on="timestamp", how="inner")
                rf_rate = joined["_rf"] if "_rf" in joined.columns else None
            else:
                rf_rate = None

        return cls(
            returns=result_df,
            rf_rate=rf_rate,
            factor_names=all_names,
            source="+".join(sources),
            frequency=factor_data_items[0].frequency,
        )
