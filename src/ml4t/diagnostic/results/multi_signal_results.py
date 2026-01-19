"""Result classes for Multi-Signal Analysis module.

This module provides Pydantic result classes for storing and serializing
multi-signal analysis outputs including summary metrics across many signals,
multiple testing corrections, and signal comparisons.

References
----------
Benjamini, Y., & Hochberg, Y. (1995). "Controlling the False Discovery Rate"
Holm, S. (1979). "A Simple Sequentially Rejective Multiple Test Procedure"
"""

from __future__ import annotations

from typing import Any

import polars as pl
from pydantic import Field

from ml4t.diagnostic.results.base import BaseResult


class MultiSignalSummary(BaseResult):
    """Summary metrics for all analyzed signals.

    Contains aggregated metrics across 50-200 signals with FDR and FWER
    corrections for multiple testing. Provides ranking, filtering, and
    DataFrame access for downstream analysis and visualization.

    Examples
    --------
    >>> summary = multi_signal_analysis.compute_summary()
    >>> print(f"Significant: {summary.n_fdr_significant}/{summary.n_signals}")
    >>> df = summary.get_dataframe()
    >>> top_signals = summary.get_significant_signals(method="fdr")
    """

    analysis_type: str = Field(default="multi_signal_summary", frozen=True)

    # ==========================================================================
    # Core Summary Data
    # ==========================================================================

    summary_data: dict[str, list[Any]] = Field(
        ...,
        description="DataFrame columns as dict of lists. Keys: column names",
    )

    # ==========================================================================
    # Metadata
    # ==========================================================================

    n_signals: int = Field(
        ...,
        ge=1,
        description="Total number of signals analyzed",
    )

    n_fdr_significant: int = Field(
        ...,
        ge=0,
        description="Number of signals significant after FDR correction",
    )

    n_fwer_significant: int = Field(
        ...,
        ge=0,
        description="Number of signals significant after FWER correction",
    )

    periods: tuple[int, ...] = Field(
        ...,
        description="Forward return periods analyzed (e.g., (1, 5, 10))",
    )

    fdr_alpha: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="FDR significance level used",
    )

    fwer_alpha: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="FWER significance level used",
    )

    # ==========================================================================
    # Correlation Data (Optional)
    # ==========================================================================

    correlation_data: dict[str, list[float]] | None = Field(
        default=None,
        description="Signal correlation matrix as dict of lists (optional)",
    )

    # ==========================================================================
    # Methods
    # ==========================================================================

    def get_dataframe(self, name: str | None = None) -> pl.DataFrame:
        """Get results as Polars DataFrame.

        Parameters
        ----------
        name : str | None
            DataFrame to retrieve:
            - None or "summary": Main summary with all signals
            - "correlation": Signal correlation matrix (if available)

        Returns
        -------
        pl.DataFrame
            Requested DataFrame
        """
        if name is None or name == "summary":
            return pl.DataFrame(self.summary_data)
        elif name == "correlation":
            if self.correlation_data is None:
                raise ValueError("Correlation data not computed")
            return pl.DataFrame(self.correlation_data)
        else:
            available = self.list_available_dataframes()
            raise ValueError(f"Unknown DataFrame '{name}'. Available: {available}")

    def list_available_dataframes(self) -> list[str]:
        """List available DataFrame views."""
        available = ["summary"]
        if self.correlation_data is not None:
            available.append("correlation")
        return available

    def get_significant_signals(
        self,
        method: str = "fdr",
    ) -> list[str]:
        """Get list of significant signal names.

        Parameters
        ----------
        method : str, default "fdr"
            Correction method: "fdr" or "fwer"

        Returns
        -------
        list[str]
            Names of significant signals
        """
        col = f"{method}_significant"
        if col not in self.summary_data:
            raise ValueError(f"Column '{col}' not in summary data")

        signal_names = self.summary_data["signal_name"]
        significant = self.summary_data[col]

        return [name for name, sig in zip(signal_names, significant) if sig]

    def get_ranking(
        self,
        metric: str = "ic_ir",
        ascending: bool = False,
        n: int | None = None,
    ) -> list[str]:
        """Get signal names ranked by metric.

        Parameters
        ----------
        metric : str, default "ic_ir"
            Metric to rank by
        ascending : bool, default False
            If True, return lowest values first
        n : int | None
            Number of signals to return (None = all)

        Returns
        -------
        list[str]
            Ranked signal names
        """
        df = self.get_dataframe()
        sorted_df = df.sort(metric, descending=not ascending)
        if n is not None:
            sorted_df = sorted_df.head(n)
        return sorted_df["signal_name"].to_list()

    def filter_signals(
        self,
        min_ic: float | None = None,
        min_ic_ir: float | None = None,
        max_turnover: float | None = None,
        significant_only: bool = False,
        significance_method: str = "fdr",
    ) -> pl.DataFrame:
        """Filter signals by criteria.

        Parameters
        ----------
        min_ic : float | None
            Minimum IC mean
        min_ic_ir : float | None
            Minimum IC IR
        max_turnover : float | None
            Maximum turnover
        significant_only : bool
            Only include significant signals
        significance_method : str
            "fdr" or "fwer" for significance filter

        Returns
        -------
        pl.DataFrame
            Filtered summary DataFrame
        """
        df = self.get_dataframe()

        if min_ic is not None and "ic_mean" in df.columns:
            df = df.filter(pl.col("ic_mean") >= min_ic)
        if min_ic_ir is not None and "ic_ir" in df.columns:
            df = df.filter(pl.col("ic_ir") >= min_ic_ir)
        if max_turnover is not None and "turnover_mean" in df.columns:
            df = df.filter(pl.col("turnover_mean") <= max_turnover)
        if significant_only:
            sig_col = f"{significance_method}_significant"
            if sig_col in df.columns:
                df = df.filter(pl.col(sig_col))

        return df

    def summary(self) -> str:
        """Get human-readable summary of results."""
        lines = [
            "=" * 60,
            "Multi-Signal Analysis Summary",
            "=" * 60,
            f"Signals Analyzed: {self.n_signals}",
            f"Periods: {self.periods}",
            "",
            "Multiple Testing Corrections:",
            f"  FDR ({self.fdr_alpha:.0%}): {self.n_fdr_significant} significant ({self.n_fdr_significant / self.n_signals:.1%})",
            f"  FWER ({self.fwer_alpha:.0%}): {self.n_fwer_significant} significant ({self.n_fwer_significant / self.n_signals:.1%})",
        ]

        # Add top signals if we have IC IR
        if "ic_ir" in self.summary_data:
            top = self.get_ranking("ic_ir", n=5)
            lines.extend(["", "Top 5 Signals by IC IR:"])
            for i, name in enumerate(top, 1):
                lines.append(f"  {i}. {name}")

        lines.append("=" * 60)
        return "\n".join(lines)


class ComparisonResult(BaseResult):
    """Detailed comparison of selected signals.

    Contains individual tear sheet data for a subset of signals
    selected for detailed comparison, along with correlation information
    and selection metadata.

    Examples
    --------
    >>> comparison = analyzer.compare(selection="uncorrelated", n=5)
    >>> for signal in comparison.signals:
    ...     tear_sheet = comparison.get_tear_sheet(signal)
    ...     print(f"{signal}: IC IR = {tear_sheet.ic_ir}")
    """

    analysis_type: str = Field(default="signal_comparison", frozen=True)

    # ==========================================================================
    # Selection Metadata
    # ==========================================================================

    signals: list[str] = Field(
        ...,
        description="Names of selected signals",
    )

    selection_method: str = Field(
        ...,
        description="Selection method used: 'top_n', 'uncorrelated', 'pareto', 'cluster', 'manual'",
    )

    selection_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters used for selection",
    )

    # ==========================================================================
    # Tear Sheet Data
    # ==========================================================================

    tear_sheets: dict[str, dict[str, Any]] = Field(
        ...,
        description="Serialized SignalTearSheet data per signal",
    )

    # ==========================================================================
    # Correlation Data
    # ==========================================================================

    correlation_matrix: dict[str, list[float]] = Field(
        ...,
        description="Pairwise correlation matrix for selected signals",
    )

    # ==========================================================================
    # Methods
    # ==========================================================================

    def get_dataframe(self, name: str | None = None) -> pl.DataFrame:
        """Get results as Polars DataFrame.

        Parameters
        ----------
        name : str | None
            DataFrame to retrieve:
            - None or "summary": Summary metrics for compared signals
            - "correlation": Correlation matrix

        Returns
        -------
        pl.DataFrame
            Requested DataFrame
        """
        if name is None or name == "summary":
            # Build summary from tear sheets
            rows = []
            for signal_name, data in self.tear_sheets.items():
                row = {"signal_name": signal_name}
                # Extract key metrics from tear sheet data
                if "ic_analysis" in data and data["ic_analysis"]:
                    ic_data = data["ic_analysis"]
                    # Get first period's metrics
                    if "ic_mean" in ic_data:
                        for period, value in ic_data["ic_mean"].items():
                            row[f"ic_mean_{period}"] = value
                            break  # Just first period for summary
                    if "ic_ir" in ic_data:
                        for period, value in ic_data["ic_ir"].items():
                            row[f"ic_ir_{period}"] = value
                            break
                rows.append(row)
            return pl.DataFrame(rows)

        elif name == "correlation":
            return pl.DataFrame(self.correlation_matrix)

        else:
            available = self.list_available_dataframes()
            raise ValueError(f"Unknown DataFrame '{name}'. Available: {available}")

    def list_available_dataframes(self) -> list[str]:
        """List available DataFrame views."""
        return ["summary", "correlation"]

    def get_tear_sheet_data(self, signal_name: str) -> dict[str, Any]:
        """Get tear sheet data for a specific signal.

        Parameters
        ----------
        signal_name : str
            Name of signal

        Returns
        -------
        dict
            Serialized tear sheet data
        """
        if signal_name not in self.tear_sheets:
            raise ValueError(f"Signal '{signal_name}' not in comparison. Available: {self.signals}")
        return self.tear_sheets[signal_name]

    def get_correlation_dataframe(self) -> pl.DataFrame:
        """Get correlation matrix as DataFrame.

        Returns
        -------
        pl.DataFrame
            Correlation matrix with signal names as columns
        """
        return pl.DataFrame(self.correlation_matrix)

    def get_pairwise_correlation(self, signal1: str, signal2: str) -> float:
        """Get correlation between two signals.

        Parameters
        ----------
        signal1 : str
            First signal name
        signal2 : str
            Second signal name

        Returns
        -------
        float
            Correlation coefficient
        """
        if signal1 not in self.correlation_matrix:
            raise ValueError(f"Signal '{signal1}' not found")
        if signal2 not in self.signals:
            raise ValueError(f"Signal '{signal2}' not found")

        idx = self.signals.index(signal2)
        return self.correlation_matrix[signal1][idx]

    def summary(self) -> str:
        """Get human-readable summary of comparison."""
        lines = [
            "=" * 60,
            "Signal Comparison",
            "=" * 60,
            f"Selection Method: {self.selection_method}",
            f"Signals Compared: {len(self.signals)}",
            "",
            "Signals:",
        ]

        for i, signal in enumerate(self.signals, 1):
            lines.append(f"  {i}. {signal}")

        if self.selection_params:
            lines.extend(["", "Selection Parameters:"])
            for key, value in self.selection_params.items():
                lines.append(f"  {key}: {value}")

        lines.append("=" * 60)
        return "\n".join(lines)
