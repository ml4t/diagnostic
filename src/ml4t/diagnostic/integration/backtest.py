"""Helpers for analyzing ml4t-backtest results from ml4t-diagnostic."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

from ml4t.diagnostic.evaluation import PortfolioAnalysis
from ml4t.diagnostic.evaluation.metrics.risk_adjusted import sharpe_ratio, sortino_ratio
from ml4t.diagnostic.evaluation.portfolio_analysis.metrics import (
    annual_return,
    calmar_ratio,
    max_drawdown,
)

if TYPE_CHECKING:
    from ml4t.backtest import BacktestResult
else:
    try:
        from ml4t.backtest import BacktestResult
        from ml4t.backtest.analytics.annualization import get_annualization_factor
    except ImportError as exc:
        raise ImportError(
            "ml4t-backtest integration requires the optional 'ml4t-backtest' package. "
            "Install with: pip install 'ml4t-diagnostic[backtest]'"
        ) from exc


def _resolve_calendar(result: BacktestResult, calendar: str | None) -> str | None:
    return calendar or (result.config.resolved_calendar if result.config else None)


def _extract_daily_frame(result: BacktestResult, calendar: str | None) -> pl.DataFrame:
    resolved_calendar = _resolve_calendar(result, calendar)
    session_aligned = result._auto_session_aligned(resolved_calendar)
    return result.to_daily_pnl(session_aligned=session_aligned)


def _get_analyzed_trades(result: BacktestResult) -> list[Any]:
    closed_trades = [
        trade for trade in result.trades if getattr(trade, "status", "closed") == "closed"
    ]
    return closed_trades if closed_trades else list(result.trades)


def _compute_trade_metrics(trades: list[Any]) -> dict[str, float]:
    if not trades:
        return {
            "num_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "avg_trade": 0.0,
            "avg_winner": 0.0,
            "avg_loser": 0.0,
            "total_fees": 0.0,
            "total_commission": 0.0,
            "total_slippage": 0.0,
        }

    wins = [trade for trade in trades if trade.pnl > 0]
    losses = [trade for trade in trades if trade.pnl < 0]
    total_wins = sum(trade.pnl for trade in wins)
    total_losses = abs(sum(trade.pnl for trade in losses))
    avg_trade = sum(trade.pnl for trade in trades) / len(trades)
    avg_winner = total_wins / len(wins) if wins else 0.0
    avg_loser = sum(trade.pnl for trade in losses) / len(losses) if losses else 0.0
    total_fees = sum(trade.fees for trade in trades)
    total_slippage = sum(trade.total_slippage_cost for trade in trades)

    return {
        "num_trades": len(trades),
        "win_rate": len(wins) / len(trades),
        "profit_factor": total_wins / total_losses if total_losses > 0 else float("inf"),
        "expectancy": avg_trade,
        "avg_trade": avg_trade,
        "avg_winner": avg_winner,
        "avg_loser": avg_loser,
        "total_fees": total_fees,
        "total_commission": total_fees,
        "total_slippage": total_slippage,
    }


def _normalize_tearsheet_metrics(
    result: BacktestResult,
    calendar: str | None,
) -> dict[str, Any]:
    metrics = dict(result.metrics)

    if "sharpe_ratio" not in metrics and "sharpe" in metrics:
        metrics["sharpe_ratio"] = metrics["sharpe"]
    if "sharpe" not in metrics and "sharpe_ratio" in metrics:
        metrics["sharpe"] = metrics["sharpe_ratio"]

    if "sortino_ratio" not in metrics and "sortino" in metrics:
        metrics["sortino_ratio"] = metrics["sortino"]
    if "sortino" not in metrics and "sortino_ratio" in metrics:
        metrics["sortino"] = metrics["sortino_ratio"]

    if "calmar_ratio" not in metrics and "calmar" in metrics:
        metrics["calmar_ratio"] = metrics["calmar"]
    if "calmar" not in metrics and "calmar_ratio" in metrics:
        metrics["calmar"] = metrics["calmar_ratio"]

    if "total_return" not in metrics and "total_return_pct" in metrics:
        metrics["total_return"] = metrics["total_return_pct"] / 100.0

    if "n_trades" not in metrics and "num_trades" in metrics:
        metrics["n_trades"] = metrics["num_trades"]
    if "num_trades" not in metrics and "n_trades" in metrics:
        metrics["num_trades"] = metrics["n_trades"]

    computed = compute_metrics_from_result(result, calendar=calendar)
    for key, value in computed.items():
        metrics.setdefault(key, value)

    if "sharpe" not in metrics and "sharpe_ratio" in metrics:
        metrics["sharpe"] = metrics["sharpe_ratio"]
    if "sortino" not in metrics and "sortino_ratio" in metrics:
        metrics["sortino"] = metrics["sortino_ratio"]
    if "calmar" not in metrics and "calmar_ratio" in metrics:
        metrics["calmar"] = metrics["calmar_ratio"]
    if "n_trades" not in metrics and "num_trades" in metrics:
        metrics["n_trades"] = metrics["num_trades"]
    if "total_commission" not in metrics and "total_fees" in metrics:
        metrics["total_commission"] = metrics["total_fees"]

    if "max_drawdown" in metrics:
        metrics["max_drawdown"] = abs(float(metrics["max_drawdown"]))

    if isinstance(metrics.get("profit_factor"), float) and math.isnan(metrics["profit_factor"]):
        metrics["profit_factor"] = 0.0

    return metrics


def portfolio_analysis_from_result(
    result: BacktestResult,
    calendar: str | None = None,
    benchmark: Any = None,
) -> PortfolioAnalysis:
    """Create a PortfolioAnalysis from a BacktestResult."""
    resolved_calendar = _resolve_calendar(result, calendar)
    periods_per_year = get_annualization_factor(resolved_calendar)
    daily_df = _extract_daily_frame(result, resolved_calendar)

    if daily_df.is_empty():
        import numpy as np

        return PortfolioAnalysis(returns=np.array([]), periods_per_year=periods_per_year)

    date_col = "date" if "date" in daily_df.columns else "session_date"
    dates = daily_df[date_col].to_list()
    returns = daily_df["return_pct"].to_numpy()

    return PortfolioAnalysis(
        returns=returns,
        dates=dates,
        benchmark=benchmark,
        periods_per_year=periods_per_year,
    )


def compute_metrics_from_result(
    result: BacktestResult,
    calendar: str | None = None,
    confidence_intervals: bool = False,
) -> dict[str, Any]:
    """Compute portfolio and trade metrics from a BacktestResult."""
    resolved_calendar = _resolve_calendar(result, calendar)
    periods_per_year = get_annualization_factor(resolved_calendar)
    daily_returns = result.to_daily_returns(calendar=resolved_calendar)
    returns_array = daily_returns.to_numpy()
    metrics: dict[str, Any] = {}

    if len(returns_array) > 0:
        sharpe = sharpe_ratio(
            returns_array,
            annualization_factor=periods_per_year,
            confidence_intervals=confidence_intervals,
        )
        if isinstance(sharpe, dict):
            metrics["sharpe_ratio"] = sharpe["sharpe"]
            metrics["sharpe_ratio_ci_lower"] = sharpe["lower_ci"]
            metrics["sharpe_ratio_ci_upper"] = sharpe["upper_ci"]
        else:
            metrics["sharpe_ratio"] = sharpe

        metrics["sortino_ratio"] = sortino_ratio(
            returns_array,
            annualization_factor=periods_per_year,
        )
        metrics["max_drawdown"] = float(max_drawdown(returns_array))
        metrics["cagr"] = float(annual_return(returns_array, periods_per_year=periods_per_year))
        metrics["calmar_ratio"] = float(calmar_ratio(returns_array, periods_per_year))
    else:
        metrics["sharpe_ratio"] = 0.0
        metrics["sortino_ratio"] = 0.0
        metrics["max_drawdown"] = 0.0
        metrics["cagr"] = 0.0
        metrics["calmar_ratio"] = 0.0

    equity_df = result.to_equity_dataframe()
    if len(equity_df) > 0:
        equity_values = equity_df["equity"].to_list()
        first_val = equity_values[0] if equity_values else 1.0
        last_val = equity_values[-1] if equity_values else 1.0
        metrics["total_return"] = float((last_val / first_val) - 1.0)
    else:
        metrics["total_return"] = 0.0

    metrics.update(_compute_trade_metrics(_get_analyzed_trades(result)))
    if result.trade_analyzer is not None:
        ta = result.trade_analyzer
        metrics["num_trades"] = ta.num_trades
        metrics["win_rate"] = ta.win_rate
        metrics["profit_factor"] = ta.profit_factor
        metrics["total_fees"] = ta.total_fees
        metrics["total_commission"] = ta.total_fees

    return metrics


def generate_tearsheet_from_result(
    result: BacktestResult,
    template: Literal["quant_trader", "hedge_fund", "risk_manager", "full"] = "full",
    theme: Literal["default", "dark", "print", "presentation"] = "default",
    title: str | None = None,
    output_path: str | Path | None = None,
    include_statistical: bool = True,
    calendar: str | None = None,
) -> str:
    """Generate an HTML tearsheet from a BacktestResult."""
    del include_statistical
    from ml4t.diagnostic.visualization.backtest import generate_backtest_tearsheet

    trades_df = result.to_trades_dataframe()
    returns = result.to_daily_returns(calendar=calendar).to_numpy()
    tearsheet_metrics = _normalize_tearsheet_metrics(result, calendar=calendar)

    if "total_pnl" not in tearsheet_metrics and result.trades:
        tearsheet_metrics["total_pnl"] = sum(trade.pnl for trade in result.trades)

    equity_df = result.to_equity_dataframe() if result.equity_curve else None

    html = generate_backtest_tearsheet(
        metrics=tearsheet_metrics,
        trades=trades_df if len(trades_df) > 0 else None,
        returns=returns if len(returns) > 0 else None,
        equity_curve=equity_df,
        template=template,
        theme=theme,
        title=title or "Backtest Tearsheet",
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html)

    return html


__all__ = [
    "compute_metrics_from_result",
    "generate_tearsheet_from_result",
    "portfolio_analysis_from_result",
]
