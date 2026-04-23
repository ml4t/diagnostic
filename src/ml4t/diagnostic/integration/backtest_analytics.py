"""Pure analytics for backtest reporting surfaces."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from ml4t.diagnostic.evaluation.portfolio_analysis.metrics import (
    annual_return,
    calmar_ratio,
    max_drawdown,
)
from ml4t.diagnostic.metrics.risk_adjusted import sharpe_ratio, sortino_ratio


def _filled_notional_expr() -> pl.Expr:
    return pl.col("quantity").abs() * pl.col("price")


def _implementation_cost_expr() -> pl.Expr:
    return pl.col("commission").fill_null(0.0) + (
        pl.col("quantity").abs() * pl.col("slippage").fill_null(0.0)
    )


def _rebalance_key_expr() -> pl.Expr:
    return (
        pl.when(pl.col("rebalance_id").is_not_null() & (pl.col("rebalance_id") != ""))
        .then(pl.col("rebalance_id"))
        .otherwise(pl.col("timestamp").dt.strftime("%Y-%m-%dT%H:%M:%S%.f"))
        .alias("rebalance_key")
    )


def _empty_turnover_timeline() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "timestamp": pl.Datetime(),
            "equity": pl.Float64(),
            "filled_notional": pl.Float64(),
            "implementation_cost": pl.Float64(),
            "turnover": pl.Float64(),
            "cost_drag": pl.Float64(),
        }
    )


def _empty_rebalance_summary() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "rebalance_key": pl.String(),
            "timestamp": pl.Datetime(),
            "filled_notional": pl.Float64(),
            "implementation_cost": pl.Float64(),
            "symbols_touched": pl.UInt32(),
        }
    )


def _empty_symbol_table() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "symbol": pl.String(),
            "net_pnl": pl.Float64(),
            "trade_count": pl.UInt32(),
            "turnover_notional": pl.Float64(),
            "implementation_cost": pl.Float64(),
            "pnl_contribution_share": pl.Float64(),
            "turnover_contribution_share": pl.Float64(),
            "cost_contribution_share": pl.Float64(),
            "persistent_negative_pnl": pl.Boolean(),
            "burden_score": pl.Float64(),
        }
    )


def _empty_drawdown_episodes() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "episode_id": pl.String(),
            "peak_timestamp": pl.Datetime(),
            "trough_timestamp": pl.Datetime(),
            "recovery_timestamp": pl.Datetime(),
            "depth": pl.Float64(),
            "peak_to_trough_bars": pl.Int64(),
            "recovery_bars": pl.Int64(),
            "status": pl.String(),
            "top_contributors": pl.String(),
        }
    )


def _align_timestamp_literal(timestamp: Any, dtype: pl.DataType | None) -> Any:
    if dtype is None or not isinstance(dtype, pl.Datetime) or not isinstance(timestamp, datetime):
        return timestamp

    time_zone = dtype.time_zone
    if time_zone is None:
        return timestamp.replace(tzinfo=None)

    target_zone = ZoneInfo(time_zone)
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=target_zone)
    return timestamp.astimezone(target_zone)


def _rolling_return(values: Sequence[float], window: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    result = np.full(array.shape[0], np.nan, dtype=float)
    if array.shape[0] < window:
        return result
    for idx in range(window - 1, array.shape[0]):
        window_values = array[idx - window + 1 : idx + 1]
        result[idx] = float(np.prod(1.0 + window_values) - 1.0)
    return result


def _rolling_sharpe(values: Sequence[float], window: int, periods_per_year: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    result = np.full(array.shape[0], np.nan, dtype=float)
    if array.shape[0] < window:
        return result
    scale = float(np.sqrt(periods_per_year))
    for idx in range(window - 1, array.shape[0]):
        window_values = array[idx - window + 1 : idx + 1]
        std = float(np.std(window_values, ddof=1))
        if std > 0:
            result[idx] = float(np.mean(window_values) / std * scale)
    return result


def compute_performance_metrics(
    daily_returns: pl.Series,
    equity_df: pl.DataFrame,
    periods_per_year: int,
    confidence_intervals: bool = False,
) -> dict[str, Any]:
    """Compute performance and stability metrics from return and equity surfaces."""
    returns_array = np.asarray(daily_returns.to_numpy(), dtype=float)
    metrics: dict[str, Any] = {}

    if returns_array.size > 0:
        sharpe = sharpe_ratio(
            returns_array,
            periods_per_year=periods_per_year,
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
            periods_per_year=periods_per_year,
        )
        metrics["max_drawdown"] = float(abs(max_drawdown(returns_array)))
        metrics["cagr"] = float(annual_return(returns_array, periods_per_year=periods_per_year))
        metrics["calmar_ratio"] = float(calmar_ratio(returns_array, periods_per_year))
        if returns_array.size > 1:
            metrics["annualized_volatility"] = float(np.std(returns_array, ddof=1)) * float(
                np.sqrt(periods_per_year)
            )
        else:
            metrics["annualized_volatility"] = 0.0
    else:
        metrics.update(
            {
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "cagr": 0.0,
                "calmar_ratio": 0.0,
                "annualized_volatility": 0.0,
            }
        )

    if equity_df.is_empty():
        total_return = 0.0
    else:
        first_equity = float(equity_df["equity"][0])
        last_equity = float(equity_df["equity"][-1])
        total_return = 0.0 if first_equity == 0 else float(last_equity / first_equity - 1.0)
    metrics["total_return"] = total_return

    rolling = pl.DataFrame(
        {
            "period_index": pl.Series("period_index", range(returns_array.size), dtype=pl.Int64),
            "rolling_sharpe_63": _rolling_sharpe(returns_array, 63, periods_per_year),
            "rolling_return_63": _rolling_return(returns_array, 63),
            "rolling_sharpe_252": _rolling_sharpe(returns_array, 252, periods_per_year),
            "rolling_return_252": _rolling_return(returns_array, 252),
        }
    )

    return {
        "metrics": metrics,
        "daily_returns": daily_returns,
        "equity_curve": equity_df,
        "rolling": rolling,
    }


def compute_edge_metrics(trades_df: pl.DataFrame) -> dict[str, Any]:
    """Compute trade-lifecycle metrics from the trade surface."""
    if trades_df.is_empty():
        return {
            "metrics": {
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
                "avg_bars_held": 0.0,
            },
            "by_symbol": pl.DataFrame(
                schema={"symbol": pl.String(), "net_pnl": pl.Float64(), "trade_count": pl.UInt32()}
            ),
        }

    closed = (
        trades_df.filter(pl.col("status") == "closed")
        if "status" in trades_df.columns
        else trades_df.clone()
    )
    if closed.is_empty():
        closed = trades_df

    wins = closed.filter(pl.col("pnl") > 0)
    losses = closed.filter(pl.col("pnl") < 0)

    total_wins = float(wins["pnl"].sum()) if not wins.is_empty() else 0.0
    total_losses = abs(float(losses["pnl"].sum())) if not losses.is_empty() else 0.0
    avg_trade = float(closed["pnl"].mean()) if closed.height else 0.0

    metrics = {
        "num_trades": closed.height,
        "win_rate": float(wins.height / closed.height) if closed.height else 0.0,
        "profit_factor": float(total_wins / total_losses) if total_losses > 0 else float("inf"),
        "expectancy": avg_trade,
        "avg_trade": avg_trade,
        "avg_winner": float(wins["pnl"].mean()) if not wins.is_empty() else 0.0,
        "avg_loser": float(losses["pnl"].mean()) if not losses.is_empty() else 0.0,
        "total_fees": float(closed["fees"].sum()) if "fees" in closed.columns else 0.0,
        "total_commission": float(closed["fees"].sum()) if "fees" in closed.columns else 0.0,
        "total_slippage": (
            float(closed["total_slippage_cost"].sum())
            if "total_slippage_cost" in closed.columns
            else 0.0
        ),
        "avg_bars_held": float(closed["bars_held"].mean())
        if "bars_held" in closed.columns
        else 0.0,
    }

    by_symbol = (
        closed.group_by("symbol")
        .agg(
            [
                pl.col("pnl").sum().alias("net_pnl"),
                pl.len().alias("trade_count"),
            ]
        )
        .sort("net_pnl", descending=True)
    )
    return {"metrics": metrics, "by_symbol": by_symbol}


def compute_activity_metrics(
    fills_df: pl.DataFrame, portfolio_state_df: pl.DataFrame
) -> dict[str, Any]:
    """Compute fill and rebalance activity metrics from raw surfaces."""
    if fills_df.is_empty():
        return {
            "metrics": {
                "num_fills": 0,
                "num_rebalance_events": 0,
                "unique_symbols_traded": 0,
                "total_filled_notional": 0.0,
                "total_implementation_cost": 0.0,
                "avg_turnover": None,
                "max_turnover": None,
            },
            "turnover_timeline": _empty_turnover_timeline(),
            "rebalance_summary": _empty_rebalance_summary(),
        }

    fills = fills_df.sort("timestamp").with_columns(
        [
            _filled_notional_expr().alias("filled_notional"),
            _implementation_cost_expr().alias("implementation_cost"),
            _rebalance_key_expr(),
        ]
    )

    rebalance_summary = (
        fills.group_by("rebalance_key")
        .agg(
            [
                pl.col("timestamp").min().alias("timestamp"),
                pl.col("filled_notional").sum().alias("filled_notional"),
                pl.col("implementation_cost").sum().alias("implementation_cost"),
                pl.col("asset").n_unique().alias("symbols_touched"),
            ]
        )
        .sort("timestamp")
    )

    turnover_timeline = _empty_turnover_timeline()
    avg_turnover: float | None = None
    max_turnover: float | None = None
    if not portfolio_state_df.is_empty():
        fills_by_timestamp = fills.group_by("timestamp").agg(
            [
                pl.col("filled_notional").sum().alias("filled_notional"),
                pl.col("implementation_cost").sum().alias("implementation_cost"),
            ]
        )
        turnover_timeline = (
            portfolio_state_df.sort("timestamp")
            .select(["timestamp", "equity"])
            .join(fills_by_timestamp, on="timestamp", how="left")
            .with_columns(
                [
                    pl.col("filled_notional").fill_null(0.0),
                    pl.col("implementation_cost").fill_null(0.0),
                ]
            )
            .with_columns(
                [
                    pl.when(pl.col("equity") > 0)
                    .then(pl.col("filled_notional") / pl.col("equity"))
                    .otherwise(0.0)
                    .alias("turnover"),
                    pl.when(pl.col("equity") > 0)
                    .then(pl.col("implementation_cost") / pl.col("equity"))
                    .otherwise(0.0)
                    .alias("cost_drag"),
                ]
            )
        )
        # Average turnover on rebalance days only (not diluted by non-trading days)
        rebalance_days = turnover_timeline.filter(pl.col("filled_notional") > 0)
        avg_turnover = (
            float(rebalance_days["turnover"].mean()) if rebalance_days.height > 0 else None
        )
        max_turnover = float(turnover_timeline["turnover"].max())

    metrics = {
        "num_fills": fills.height,
        "num_rebalance_events": rebalance_summary.height,
        "unique_symbols_traded": int(fills["asset"].n_unique()),
        "total_filled_notional": float(fills["filled_notional"].sum()),
        "total_implementation_cost": float(fills["implementation_cost"].sum()),
        "avg_turnover": avg_turnover,
        "max_turnover": max_turnover,
    }
    return {
        "metrics": metrics,
        "turnover_timeline": turnover_timeline,
        "rebalance_summary": rebalance_summary,
    }


def compute_occupancy_metrics(portfolio_state_df: pl.DataFrame) -> dict[str, Any]:
    """Compute occupancy metrics from the portfolio-state surface."""
    if portfolio_state_df.is_empty():
        return {
            "metrics": {
                "time_in_market": 0.0,
                "avg_invested_fraction": 0.0,
                "avg_gross_exposure": 0.0,
                "max_gross_exposure": 0.0,
                "avg_net_exposure": 0.0,
                "max_net_exposure": 0.0,
                "avg_open_positions": 0.0,
                "max_open_positions": 0,
            },
            "timeline": pl.DataFrame(
                schema={
                    "timestamp": pl.Datetime(),
                    "equity": pl.Float64(),
                    "gross_exposure": pl.Float64(),
                    "net_exposure": pl.Float64(),
                    "open_positions": pl.Int32(),
                    "invested_fraction": pl.Float64(),
                    "gross_exposure_fraction": pl.Float64(),
                    "net_exposure_fraction": pl.Float64(),
                }
            ),
        }

    timeline = portfolio_state_df.sort("timestamp").with_columns(
        [
            pl.when(pl.col("equity") > 0)
            .then(pl.col("gross_exposure") / pl.col("equity"))
            .otherwise(0.0)
            .alias("invested_fraction"),
            pl.when(pl.col("equity") > 0)
            .then(pl.col("gross_exposure") / pl.col("equity"))
            .otherwise(0.0)
            .alias("gross_exposure_fraction"),
            pl.when(pl.col("equity") > 0)
            .then(pl.col("net_exposure") / pl.col("equity"))
            .otherwise(0.0)
            .alias("net_exposure_fraction"),
        ]
    )

    metrics = {
        "time_in_market": float(
            timeline.select((pl.col("open_positions") > 0).mean().alias("value"))["value"][0]
        ),
        "avg_invested_fraction": float(timeline["invested_fraction"].mean()),
        "avg_gross_exposure": float(timeline["gross_exposure_fraction"].mean()),
        "max_gross_exposure": float(timeline["gross_exposure_fraction"].max()),
        "avg_net_exposure": float(timeline["net_exposure_fraction"].mean()),
        "max_net_exposure": float(timeline["net_exposure_fraction"].abs().max()),
        "avg_open_positions": float(timeline["open_positions"].mean()),
        "max_open_positions": int(timeline["open_positions"].max()),
    }
    return {"metrics": metrics, "timeline": timeline}


def compute_attribution_metrics(trades_df: pl.DataFrame, fills_df: pl.DataFrame) -> dict[str, Any]:
    """Compute symbol contribution and burden metrics from trades and fills."""
    trade_group = (
        trades_df.group_by("symbol").agg(
            [
                pl.col("pnl").sum().alias("net_pnl"),
                pl.len().alias("trade_count"),
            ]
        )
        if not trades_df.is_empty()
        else pl.DataFrame(
            schema={"symbol": pl.String(), "net_pnl": pl.Float64(), "trade_count": pl.UInt32()}
        )
    )

    fill_group = (
        fills_df.with_columns(
            [
                _filled_notional_expr().alias("turnover_notional"),
                _implementation_cost_expr().alias("implementation_cost"),
            ]
        )
        .group_by("asset")
        .agg(
            [
                pl.col("turnover_notional").sum().alias("turnover_notional"),
                pl.col("implementation_cost").sum().alias("implementation_cost"),
            ]
        )
        .rename({"asset": "symbol"})
        if not fills_df.is_empty()
        else pl.DataFrame(
            schema={
                "symbol": pl.String(),
                "turnover_notional": pl.Float64(),
                "implementation_cost": pl.Float64(),
            }
        )
    )

    if trade_group.is_empty() and fill_group.is_empty():
        return {
            "by_symbol": _empty_symbol_table(),
            "top_contributors": _empty_symbol_table(),
            "top_drags": _empty_symbol_table(),
            "metrics": {
                "top_5_pnl_share": None,
                "top_5_turnover_share": None,
                "top_5_cost_share": None,
            },
        }

    symbols = pl.concat(
        [
            trade_group.select("symbol"),
            fill_group.select("symbol"),
        ],
        how="vertical_relaxed",
    ).unique()

    by_symbol = (
        symbols.join(trade_group, on="symbol", how="left")
        .join(fill_group, on="symbol", how="left")
        .with_columns(
            [
                pl.col("net_pnl").fill_null(0.0),
                pl.col("trade_count").fill_null(0),
                pl.col("turnover_notional").fill_null(0.0),
                pl.col("implementation_cost").fill_null(0.0),
            ]
        )
    )

    total_pnl = float(by_symbol["net_pnl"].sum())
    total_turnover = float(by_symbol["turnover_notional"].sum())
    total_cost = float(by_symbol["implementation_cost"].sum())

    pnl_share = (
        (pl.col("net_pnl") / total_pnl) if total_pnl != 0 else pl.lit(None, dtype=pl.Float64())
    )
    turnover_share = (
        (pl.col("turnover_notional") / total_turnover)
        if total_turnover > 0
        else pl.lit(None, dtype=pl.Float64())
    )
    cost_share = (
        (pl.col("implementation_cost") / total_cost)
        if total_cost > 0
        else pl.lit(None, dtype=pl.Float64())
    )

    by_symbol = (
        by_symbol.with_columns(
            [
                pnl_share.alias("pnl_contribution_share"),
                turnover_share.alias("turnover_contribution_share"),
                cost_share.alias("cost_contribution_share"),
                (pl.col("net_pnl") < 0).alias("persistent_negative_pnl"),
            ]
        )
        .with_columns(
            (
                pl.col("turnover_contribution_share").fill_null(0.0)
                + pl.col("cost_contribution_share").fill_null(0.0)
                + pl.when(pl.col("persistent_negative_pnl")).then(0.5).otherwise(0.0)
            ).alias("burden_score")
        )
        .sort("net_pnl", descending=True)
    )

    top_contributors = by_symbol.sort("net_pnl", descending=True).head(5)
    top_drags = by_symbol.sort("net_pnl").head(5)

    metrics = {
        "top_5_pnl_share": (
            float(top_contributors["pnl_contribution_share"].sum())
            if total_pnl != 0 and not top_contributors.is_empty()
            else None
        ),
        "top_5_turnover_share": (
            float(top_contributors["turnover_contribution_share"].fill_null(0.0).sum())
            if total_turnover > 0 and not top_contributors.is_empty()
            else None
        ),
        "top_5_cost_share": (
            float(top_contributors["cost_contribution_share"].fill_null(0.0).sum())
            if total_cost > 0 and not top_contributors.is_empty()
            else None
        ),
    }
    return {
        "by_symbol": by_symbol,
        "top_contributors": top_contributors,
        "top_drags": top_drags,
        "metrics": metrics,
    }


def compute_drawdown_anatomy(equity_df: pl.DataFrame, trades_df: pl.DataFrame) -> dict[str, Any]:
    """Compute drawdown episodes and peak-to-trough contributors."""
    if equity_df.is_empty():
        return {
            "metrics": {"current_drawdown": 0.0, "max_drawdown": 0.0, "num_drawdowns": 0},
            "episodes": _empty_drawdown_episodes(),
            "contributors": {},
        }

    drawdowns = equity_df["drawdown"].to_list()
    timestamps = equity_df["timestamp"].to_list()
    periods: list[dict[str, Any]] = []
    in_drawdown = False
    peak_idx = 0
    trough_idx = 0
    trough_depth = 0.0

    for idx, drawdown in enumerate(drawdowns):
        value = float(drawdown)
        if value < 0 and not in_drawdown:
            in_drawdown = True
            peak_idx = idx - 1 if idx > 0 else 0
            trough_idx = idx
            trough_depth = value
        elif in_drawdown:
            if value < trough_depth:
                trough_idx = idx
                trough_depth = value
            elif value >= 0:
                periods.append(
                    {
                        "peak_idx": peak_idx,
                        "trough_idx": trough_idx,
                        "recovery_idx": idx,
                        "depth": trough_depth,
                    }
                )
                in_drawdown = False

    if in_drawdown:
        periods.append(
            {
                "peak_idx": peak_idx,
                "trough_idx": trough_idx,
                "recovery_idx": None,
                "depth": trough_depth,
            }
        )

    periods.sort(key=lambda period: period["depth"])
    top_periods = periods[:5]
    contributors: dict[str, pl.DataFrame] = {}
    rows: list[dict[str, Any]] = []

    for idx, period in enumerate(top_periods, start=1):
        episode_id = f"drawdown_{idx}"
        peak_timestamp = timestamps[period["peak_idx"]]
        trough_timestamp = timestamps[period["trough_idx"]]
        exit_time_dtype = trades_df.schema.get("exit_time")
        aligned_peak_timestamp = _align_timestamp_literal(peak_timestamp, exit_time_dtype)
        aligned_trough_timestamp = _align_timestamp_literal(trough_timestamp, exit_time_dtype)
        recovery_idx = period["recovery_idx"]
        recovery_timestamp = timestamps[recovery_idx] if isinstance(recovery_idx, int) else None
        contributor_frame = pl.DataFrame(schema={"symbol": pl.String(), "pnl": pl.Float64()})
        contributor_labels = ""

        if not trades_df.is_empty():
            contributor_frame = (
                trades_df.filter(
                    (pl.col("exit_time") >= aligned_peak_timestamp)
                    & (pl.col("exit_time") <= aligned_trough_timestamp)
                )
                .group_by("symbol")
                .agg(pl.col("pnl").sum().alias("pnl"))
                .filter(pl.col("pnl") < 0)
                .sort("pnl")
                .head(3)
            )
            if not contributor_frame.is_empty():
                contributor_labels = ", ".join(contributor_frame["symbol"].to_list())
        contributors[episode_id] = contributor_frame
        rows.append(
            {
                "episode_id": episode_id,
                "peak_timestamp": peak_timestamp,
                "trough_timestamp": trough_timestamp,
                "recovery_timestamp": recovery_timestamp,
                "depth": float(period["depth"]),
                "peak_to_trough_bars": period["trough_idx"] - period["peak_idx"],
                "recovery_bars": (
                    int(recovery_idx - period["trough_idx"])
                    if isinstance(recovery_idx, int)
                    else None
                ),
                "status": "recovered" if recovery_timestamp is not None else "ongoing",
                "top_contributors": contributor_labels,
            }
        )

    return {
        "metrics": {
            "current_drawdown": float(drawdowns[-1]),
            "max_drawdown": float(min(drawdowns)),
            "num_drawdowns": len(periods),
        },
        "episodes": pl.DataFrame(rows) if rows else _empty_drawdown_episodes(),
        "contributors": contributors,
    }
