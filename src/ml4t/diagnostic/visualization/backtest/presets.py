"""Preset configuration for backtest dashboard rendering.

Presets replace divergent tearsheet personalities with lightweight display
preferences over a shared dashboard architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PresetName = Literal["quant_trader", "hedge_fund", "risk_manager", "full"]


@dataclass(frozen=True)
class BacktestDashboardPreset:
    """Display preset for the integrated backtest dashboard."""

    name: PresetName
    hero_metrics: tuple[str, ...]
    workspace_order: tuple[str, ...]


_PRESETS: dict[PresetName, BacktestDashboardPreset] = {
    "quant_trader": BacktestDashboardPreset(
        name="quant_trader",
        hero_metrics=(
            "sharpe_ratio",
            "win_rate",
            "profit_factor",
            "avg_trade",
            "n_trades",
            "avg_turnover",
        ),
        workspace_order=("overview", "trading", "performance", "validation", "ml", "factors"),
    ),
    "hedge_fund": BacktestDashboardPreset(
        name="hedge_fund",
        hero_metrics=(
            "sharpe_ratio",
            "cagr",
            "max_drawdown",
            "calmar_ratio",
            "total_commission",
            "total_slippage",
        ),
        workspace_order=("overview", "performance", "trading", "validation", "factors", "ml"),
    ),
    "risk_manager": BacktestDashboardPreset(
        name="risk_manager",
        hero_metrics=(
            "sharpe_ratio",
            "max_drawdown",
            "cagr",
            "dsr_probability",
            "min_trl",
            "profit_factor",
        ),
        workspace_order=("overview", "validation", "performance", "trading", "factors", "ml"),
    ),
    "full": BacktestDashboardPreset(
        name="full",
        hero_metrics=(
            "total_return",
            "cagr",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
            "profit_factor",
        ),
        workspace_order=("overview", "performance", "trading", "validation", "factors", "ml"),
    ),
}


def get_dashboard_preset(name: PresetName) -> BacktestDashboardPreset:
    """Return a dashboard preset by name."""
    return _PRESETS[name]
