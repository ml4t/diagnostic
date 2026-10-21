"""Round-trip report tables over existing cash statistics and explicit metric samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
import polars as pl

PNL_METRICS = (
    "total_pnl",
    "gross_profit",
    "gross_loss",
    "profit_factor",
    "avg_pnl",
    "avg_winner",
    "avg_loser",
    "payoff_ratio",
    "max_pnl",
    "min_pnl",
)
SUMMARY_METRICS = ("n_trades", "win_rate", "n_winners", "n_losers", "n_even")
DURATION_METRICS = ("mean", "median", "max", "min")
RETURN_METRICS = (
    "mean",
    "mean_winner",
    "mean_loser",
    "median",
    "median_winner",
    "median_loser",
    "max",
    "min",
)


def _sample_statistic(values: pd.Series, operation: str) -> Any:
    known = values.dropna()
    return getattr(known, operation)() if len(known) else None


@dataclass
class RoundTripStatisticsResult:
    """Long-form native tables plus exact sample coverage and selected definitions.

    Cash/return tables use Float64 values; duration values retain typed ns.
    Scope identifies all/long/short/asset observations. Asset IDs remain values,
    not coerced column names. Compatibility/rendering projects these same tables.
    """

    tables: dict[str, pl.DataFrame]
    coverage: pl.DataFrame
    metadata: dict[str, Any]


def summarize_round_trips(
    observations: pl.DataFrame,
    *,
    pnl_col: str = "net_pnl",
    return_col: str = "net_nav_contribution",
    duration_col: str = "duration",
    method: Literal["native", "legacy_pyfolio"] = "native",
) -> RoundTripStatisticsResult:
    """Build cash/count/duration/return/asset tables with per-metric missingness.

    Native cash observations must be known/finite; return and duration may be
    undefined. The explicit legacy profile retains independent NaN masks, IEEE
    return arrays, profitable/all-row versus count/nonmissing-PnL populations and
    zero profit factor for all-loss samples. Native profit-factor defaults stay
    unchanged. No execution prices, NAV or calendar durations are invented.
    """
    from ml4t.diagnostic.evaluation.trade_analysis import TradeStatistics

    from .holdings import _numeric

    if method not in ("native", "legacy_pyfolio"):
        raise ValueError("method must be native or explicit legacy_pyfolio")
    if not isinstance(observations, pl.DataFrame):
        raise TypeError("round-trip observations must be a Polars DataFrame")
    required = {"asset", "direction", pnl_col, return_col, duration_col}
    if missing := required.difference(observations.columns):
        raise ValueError(f"round-trip statistics require columns: {sorted(missing)}")
    frame = _numeric(observations.clone(), pnl_col, nullable=method == "legacy_pyfolio")
    asset_dtype = frame["asset"].dtype
    if asset_dtype != pl.String and not asset_dtype.is_integer() and asset_dtype != pl.Null:
        raise TypeError("round-trip asset IDs must be stable strings or integers")
    if (
        frame["direction"].drop_nulls().len()
        and not frame["direction"].drop_nulls().is_in(["long", "short"]).all()
    ):
        raise ValueError("round-trip direction must be long/short or explicitly unknown")
    if method == "native" and (frame["asset"].null_count() or frame["direction"].null_count()):
        raise ValueError("native round-trip identity/direction must be known")
    returns = frame[return_col]
    if returns.null_count() != frame.height and not returns.dtype.is_numeric():
        raise TypeError("selected round-trip returns must be real numeric fractions")
    frame = frame.with_columns(pl.col(return_col).cast(pl.Float64).fill_nan(None))
    if method == "native" and not frame[return_col].drop_nulls().is_finite().all():
        raise ValueError("native selected returns must be finite or explicitly undefined")
    duration = frame[duration_col]
    if duration.dtype != pl.Null and not isinstance(duration.dtype, pl.Duration):
        raise TypeError(
            "round-trip duration must have an explicit temporal unit, not bare integers"
        )
    refined = duration.cast(pl.Duration("ns"))
    if duration.dtype != pl.Null and not refined.cast(duration.dtype).equals(duration):
        raise ValueError("round-trip duration cannot be represented losslessly in nanoseconds")
    frame = frame.with_columns(refined.alias(duration_col))
    if method == "native" and (refined.drop_nulls().cast(pl.Int64) < 0).any():
        raise ValueError("native elapsed durations cannot be negative")
    scopes = [("all", None, frame)] if frame.height else []
    for side in ("short", "long"):
        selected = frame.filter(pl.col("direction") == side)
        if selected.height:
            scopes.append((side, None, selected))
    for asset in frame["asset"].drop_nulls().unique().sort().to_list():
        scopes.append(("asset", asset, frame.filter(pl.col("asset") == asset)))
    rows: dict[str, list[dict[str, Any]]] = {
        key: [] for key in ("pnl", "summary", "duration", "returns", "symbols")
    }
    coverage = []
    for scope, asset, part in scopes:
        cash = part.filter(pl.col(pnl_col).is_not_null())
        duration_values = part[duration_col].to_pandas()
        return_values = part[return_col].to_pandas()
        coverage.append(
            {
                "scope": scope,
                "asset": asset,
                "n_rows": part.height,
                "n_known_pnl": cash.height,
                "n_known_duration": part.height - part[duration_col].null_count(),
                "n_known_returns": part.height - part[return_col].null_count(),
                "n_finite_returns": part[return_col].is_finite().sum(),
                "n_negative_durations": (part[duration_col].cast(pl.Int64) < 0).sum(),
            }
        )
        if scope != "asset":
            stats = (
                TradeStatistics.from_frame(cash.select(pl.col(pnl_col).alias("pnl")))
                if cash.height
                else None
            )
            profit_factor = stats.profit_factor if stats is not None else None
            if method == "legacy_pyfolio" and stats is not None and stats.gross_loss:
                assert stats.gross_profit is not None
                profit_factor = stats.gross_profit / abs(stats.gross_loss)
            pnl_values = {
                "total_pnl": stats.total_pnl if stats else 0.0,
                "gross_profit": stats.gross_profit if stats else 0.0,
                "gross_loss": stats.gross_loss if stats else 0.0,
                "profit_factor": profit_factor,
                "avg_pnl": stats.avg_pnl if stats else None,
                "avg_winner": stats.avg_winner if stats else None,
                "avg_loser": stats.avg_loser if stats else None,
                "payoff_ratio": stats.avg_winner / abs(stats.avg_loser)
                if stats and stats.avg_winner is not None and stats.avg_loser
                else None,
                "max_pnl": stats.max_pnl if stats else None,
                "min_pnl": stats.min_pnl if stats else None,
            }
            winners = stats.n_winners if stats else 0
            summary_values = {
                "n_trades": cash.height,
                "win_rate": winners / part.height
                if method == "legacy_pyfolio"
                else stats.win_rate
                if stats
                else None,
                "n_winners": winners,
                "n_losers": stats.n_losers if stats else 0,
                "n_even": part.filter(pl.col(pnl_col) == 0).height,
            }
            for key, values in (("pnl", pnl_values), ("summary", summary_values)):
                rows[key].extend(
                    {"scope": scope, "asset": asset, "metric": metric, "value": value}
                    for metric, value in values.items()
                )
            for metric in DURATION_METRICS:
                value = _sample_statistic(duration_values, metric)
                rows["duration"].append(
                    {
                        "scope": scope,
                        "asset": asset,
                        "metric": metric,
                        "value": None if pd.isna(value) else value.value,
                    }
                )
        positives = return_values[return_values > 0]
        negatives = return_values[return_values < 0]
        return_stats = {
            name: _sample_statistic(values, operation)
            for name, values, operation in (
                ("mean", return_values, "mean"),
                ("mean_winner", positives, "mean"),
                ("mean_loser", negatives, "mean"),
                ("median", return_values, "median"),
                ("median_winner", positives, "median"),
                ("median_loser", negatives, "median"),
                ("max", return_values, "max"),
                ("min", return_values, "min"),
            )
        }
        key = "symbols" if scope == "asset" else "returns"
        rows[key].extend(
            {
                "scope": scope,
                "asset": asset,
                "metric": metric,
                "value": None if pd.isna(value) else float(value),
            }
            for metric, value in return_stats.items()
        )
    tables = {}
    for key, records in rows.items():
        table = pl.DataFrame(
            records,
            schema={
                "scope": pl.String,
                "asset": asset_dtype,
                "metric": pl.String,
                "value": pl.Int64 if key == "duration" else pl.Float64,
            },
        )
        if key == "duration":
            table = table.with_columns(pl.col("value").cast(pl.Duration("ns")))
        tables[key] = table
    return RoundTripStatisticsResult(
        tables,
        pl.DataFrame(
            coverage,
            schema={
                "scope": pl.String,
                "asset": asset_dtype,
                **dict.fromkeys(
                    (
                        "n_rows",
                        "n_known_pnl",
                        "n_known_duration",
                        "n_known_returns",
                        "n_finite_returns",
                        "n_negative_durations",
                    ),
                    pl.UInt64,
                ),
            },
        ),
        {
            "method": method,
            "pnl_col": pnl_col,
            "return_col": return_col,
            "duration_unit": "ns",
            "sample_state": "observed" if frame.height else "empty",
            "summary_population": "percent profitable/all rows; count/nonmissing PnL"
            if method == "legacy_pyfolio"
            else "known finite PnL",
            "return_sample": "metric-specific signs/missingness; not PnL winner masks",
            "duration_sample": "known typed elapsed durations, not fabricated zero days",
        },
    )
