"""IR_tc (Transaction-Cost Adjusted Information Ratio) result classes.

This module provides result classes for storing transaction-cost-adjusted
Information Ratio analysis outputs.

References
----------
Lopez de Prado, M. (2018). "Advances in Financial Machine Learning"
"""

from __future__ import annotations

import polars as pl
from pydantic import Field

from ml4t.diagnostic.results.base import BaseResult


class IRtcResult(BaseResult):
    """Results from transaction-cost-adjusted Information Ratio analysis.

    IR_tc measures the risk-adjusted IC after accounting for the cost
    of turnover required to maintain the signal-based portfolio.

    IR_tc = (IC * spread_return - turnover * cost) / volatility

    Examples
    --------
    >>> result = ir_tc_result
    >>> print(result.summary())
    """

    analysis_type: str = Field(default="ir_tc_analysis", frozen=True)

    # ==========================================================================
    # Configuration
    # ==========================================================================

    cost_per_trade: float = Field(
        ...,
        description="Transaction cost per unit turnover used",
    )

    # ==========================================================================
    # Results by Period
    # ==========================================================================

    ir_gross: dict[str, float] = Field(
        ...,
        description="Gross IR (before transaction costs) per period",
    )

    ir_tc: dict[str, float] = Field(
        ...,
        description="Net IR (after transaction costs) per period",
    )

    implied_cost: dict[str, float] = Field(
        ...,
        description="Implied cost from turnover per period",
    )

    breakeven_cost: dict[str, float] = Field(
        ...,
        description="Breakeven cost (cost at which IR_tc = 0)",
    )

    cost_drag: dict[str, float] = Field(
        ...,
        description="Percentage of gross return lost to costs",
    )

    def get_dataframe(self, name: str | None = None) -> pl.DataFrame:
        """Get results as Polars DataFrame."""
        periods = list(self.ir_gross.keys())
        return pl.DataFrame(
            {
                "period": periods,
                "ir_gross": [self.ir_gross[p] for p in periods],
                "ir_tc": [self.ir_tc[p] for p in periods],
                "implied_cost": [self.implied_cost[p] for p in periods],
                "breakeven_cost": [self.breakeven_cost[p] for p in periods],
                "cost_drag": [self.cost_drag[p] for p in periods],
            }
        )

    def list_available_dataframes(self) -> list[str]:
        """List available DataFrame views."""
        return ["primary"]

    def summary(self) -> str:
        """Get human-readable summary of IR_tc results."""
        lines = [
            "=" * 60,
            "Transaction-Cost Adjusted IR Summary",
            "=" * 60,
            "",
            f"Cost per Trade: {self.cost_per_trade:.4f} ({self.cost_per_trade * 10000:.0f} bps)",
            "",
            "Period       IR_gross    IR_tc    Cost Drag   Breakeven",
            "-" * 60,
        ]

        for period in self.ir_gross:
            lines.append(
                f"{period:<12} {self.ir_gross[period]:>8.4f}  {self.ir_tc[period]:>8.4f}  "
                f"{self.cost_drag[period]:>8.1%}  {self.breakeven_cost[period]:>8.4f}"
            )

        return "\n".join(lines)
