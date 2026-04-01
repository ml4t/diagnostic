"""Helpers for analyzing ml4t-backtest results from ml4t-diagnostic."""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Sequence
from datetime import date, datetime, time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

from ml4t.diagnostic.evaluation import PortfolioAnalysis
from ml4t.diagnostic.integration.backtest_profile import (
    BacktestProfile,
)
from ml4t.diagnostic.integration.backtest_profile import (
    analyze_backtest_result as _analyze_backtest_result,
)
from ml4t.diagnostic.integration.report_metadata import BacktestReportMetadata

if TYPE_CHECKING:
    from ml4t.backtest import BacktestConfig, BacktestResult
    from ml4t.backtest.types import Trade
else:
    try:
        from ml4t.backtest import BacktestConfig, BacktestResult
        from ml4t.backtest.analytics.annualization import get_annualization_factor
        from ml4t.backtest.types import Fill, OrderSide, Trade
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
    profile = analyze_backtest_result(
        result=result,
        calendar=calendar,
        confidence_intervals=confidence_intervals,
    )
    metrics = dict(profile.summary)
    if result.trade_analyzer is not None:
        ta = result.trade_analyzer
        metrics["num_trades"] = ta.num_trades
        metrics["win_rate"] = ta.win_rate
        metrics["profit_factor"] = ta.profit_factor
        metrics["total_fees"] = ta.total_fees
        metrics["total_commission"] = ta.total_fees

    return metrics


def analyze_backtest_result(
    result: BacktestResult,
    calendar: str | None = None,
    benchmark: Any = None,
    confidence_intervals: bool = False,
) -> BacktestProfile:
    """Create a lazy BacktestProfile from a BacktestResult."""
    return _analyze_backtest_result(
        result=result,
        calendar=calendar,
        benchmark=benchmark,
        confidence_intervals=confidence_intervals,
    )


def profile_from_run_artifacts(
    backtest_dir: str | Path,
    *,
    predictions_path: str | Path | None = None,
    signals_path: str | Path | None = None,
    calendar: str | None = None,
    benchmark: Any = None,
    confidence_intervals: bool = False,
) -> BacktestProfile:
    """Build a BacktestProfile from case-study run artifacts."""
    artifact_dir = Path(backtest_dir)
    trades_df = pl.read_parquet(artifact_dir / "trades.parquet")
    daily_returns_df = pl.read_parquet(artifact_dir / "daily_returns.parquet")
    weights_df = _load_weights_dataframe(_default_signals_artifact_path(artifact_dir))
    spec = _load_artifact_spec(artifact_dir)
    config = _load_artifact_config(spec)
    feed_spec = config.resolved_feed_spec if config is not None else None
    weights_df = _normalize_surface_from_feed_spec(weights_df, feed_spec)

    active_start = _infer_active_start_timestamp(
        daily_returns_df=daily_returns_df,
        trades_df=trades_df,
        weights_df=weights_df,
    )
    if active_start is not None:
        daily_returns_df = _trim_dataframe_from_timestamp(
            daily_returns_df,
            timestamp_col=_first_present_column(daily_returns_df, ("timestamp", "date")),
            start=active_start,
        )
        trades_df = _trim_dataframe_from_timestamp(
            trades_df,
            timestamp_col=_first_present_column(trades_df, ("entry_time", "timestamp")),
            start=active_start,
        )
        weights_df = _trim_dataframe_from_timestamp(
            weights_df,
            timestamp_col=_first_present_column(weights_df, ("timestamp", "date")),
            start=active_start,
        )

    # Prefer real artifacts when available; fall back to reconstruction.
    # Track provenance so the tearsheet can show data-source footnotes.
    data_sources: dict[str, str] = {}

    equity_path = artifact_dir / "equity.parquet"
    if equity_path.exists():
        equity_curve = _equity_curve_from_parquet(equity_path)
        data_sources["equity"] = "artifact"
    else:
        equity_curve = _equity_curve_from_daily_returns(daily_returns_df, spec)
        data_sources["equity"] = "reconstructed from daily returns"

    portfolio_state_path = artifact_dir / "portfolio_state.parquet"
    if portfolio_state_path.exists():
        portfolio_state = _portfolio_state_from_parquet(portfolio_state_path)
        data_sources["portfolio_state"] = "artifact"
    else:
        portfolio_state = _portfolio_state_from_weights(weights_df, equity_curve)
        data_sources["portfolio_state"] = "reconstructed from weights"

    fills_path = artifact_dir / "fills.parquet"
    if fills_path.exists():
        fills = _fills_from_parquet(fills_path)
        data_sources["fills"] = "artifact"
    else:
        fills = _fills_from_weights(weights_df, equity_curve)
        data_sources["fills"] = "reconstructed from weights (no real execution data)"

    result = BacktestResult(
        trades=_trades_from_dataframe(trades_df),
        equity_curve=equity_curve,
        fills=fills,
        metrics={},
        config=config,
        portfolio_state=portfolio_state,
    )

    resolved_predictions_path = (
        Path(predictions_path)
        if predictions_path is not None
        else _resolve_prediction_artifact_path(artifact_dir, spec)
    )
    if signals_path is not None:
        weights_df = _normalize_surface_from_feed_spec(
            _load_weights_dataframe(Path(signals_path)),
            feed_spec,
        )

    predictions_df = (
        _normalize_surface_from_feed_spec(pl.read_parquet(resolved_predictions_path), feed_spec)
        if resolved_predictions_path is not None and resolved_predictions_path.exists()
        else None
    )
    if active_start is not None and predictions_df is not None:
        predictions_df = _trim_dataframe_from_timestamp(
            predictions_df,
            timestamp_col=_first_present_column(predictions_df, ("timestamp", "date", "session_date")),
            start=active_start,
        )
    signals_df = _load_weights_as_signal_surface_from_dataframe(weights_df)
    if not signals_df.is_empty() and "signal_value" in signals_df.columns:
        data_sources["signals"] = "derived from portfolio weights (not model signals)"

    return BacktestProfile(
        result=result,
        calendar=calendar,
        benchmark=benchmark,
        confidence_intervals=confidence_intervals,
        predictions_override=predictions_df,
        signals_override=signals_df,
        strategy_metadata_override=_build_strategy_metadata(spec, predictions_df, artifact_dir),
        data_sources=data_sources,
    )


def generate_tearsheet_from_result(
    result: BacktestResult,
    template: Literal["quant_trader", "hedge_fund", "risk_manager", "full"] = "full",
    theme: Literal["default", "dark", "print", "presentation"] = "default",
    title: str | None = None,
    output_path: str | Path | None = None,
    include_statistical: bool = True,
    calendar: str | None = None,
    benchmark: Any = None,
    benchmark_name: str = "Benchmark",
    report_metadata: BacktestReportMetadata | None = None,
) -> str:
    """Generate an HTML tearsheet from a BacktestResult."""
    del include_statistical
    from ml4t.diagnostic.visualization.backtest import generate_backtest_tearsheet

    profile = analyze_backtest_result(result, calendar=calendar, benchmark=benchmark)
    trades_df = result.to_trades_dataframe()
    returns = result.to_daily_returns(calendar=calendar).to_numpy()
    tearsheet_metrics = _normalize_tearsheet_metrics(result, calendar=calendar)

    if "total_pnl" not in tearsheet_metrics and result.trades:
        tearsheet_metrics["total_pnl"] = sum(trade.pnl for trade in result.trades)

    equity_df = result.to_equity_dataframe() if result.equity_curve else None

    html = generate_backtest_tearsheet(
        profile=profile,
        metrics=tearsheet_metrics,
        trades=trades_df if len(trades_df) > 0 else None,
        returns=returns if len(returns) > 0 else None,
        equity_curve=equity_df,
        template=template,
        theme=theme,
        title=title,
        benchmark_returns=benchmark,
        benchmark_name=benchmark_name,
        report_metadata=report_metadata,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html)

    return html


def generate_tearsheet_from_run_artifacts(
    backtest_dir: str | Path,
    *,
    predictions_path: str | Path | None = None,
    signals_path: str | Path | None = None,
    template: Literal["quant_trader", "hedge_fund", "risk_manager", "full"] = "full",
    theme: Literal["default", "dark", "print", "presentation"] = "default",
    output_path: str | Path | None = None,
    calendar: str | None = None,
    benchmark: Any = None,
    benchmark_name: str = "Benchmark",
    report_metadata: BacktestReportMetadata | None = None,
) -> str:
    """Generate a tearsheet directly from case-study run artifacts."""
    from ml4t.diagnostic.visualization.backtest import generate_backtest_tearsheet

    profile = profile_from_run_artifacts(
        backtest_dir=backtest_dir,
        predictions_path=predictions_path,
        signals_path=signals_path,
        calendar=calendar,
        benchmark=benchmark,
    )
    effective_metadata = report_metadata or _build_report_metadata_from_artifacts(
        Path(backtest_dir),
        profile.strategy_metadata,
    )

    html = generate_backtest_tearsheet(
        profile=profile,
        template=template,
        theme=theme,
        benchmark_returns=benchmark,
        benchmark_name=benchmark_name,
        report_metadata=effective_metadata,
        output_path=output_path,
    )
    return html


def _load_artifact_spec(backtest_dir: Path) -> dict[str, Any]:
    spec_path = backtest_dir / "spec.json"
    if not spec_path.exists():
        return {}
    return json.loads(spec_path.read_text())


def _load_artifact_config(spec: dict[str, Any]) -> BacktestConfig | None:
    config_data = spec.get("backtest_config")
    if not isinstance(config_data, dict):
        return None
    try:
        return BacktestConfig.from_dict(config_data)
    except Exception as exc:
        import warnings

        warnings.warn(
            f"BacktestConfig parsing failed: {exc}. "
            "Initial capital, calendar, and trading hours will use defaults.",
            stacklevel=2,
        )
        return None


def _default_signals_artifact_path(backtest_dir: Path) -> Path | None:
    weights_path = backtest_dir / "weights.parquet"
    return weights_path if weights_path.exists() else None


def _load_weights_dataframe(weights_path: Path | None) -> pl.DataFrame:
    if weights_path is None or not weights_path.exists():
        return pl.DataFrame()
    return pl.read_parquet(weights_path)


def _resolve_prediction_artifact_path(backtest_dir: Path, spec: dict[str, Any]) -> Path | None:
    prediction_hash = (
        spec.get("backtest_config", {})
        .get("metadata", {})
        .get("prediction_hash")
    )
    run_log_dir = backtest_dir.parent.parent if backtest_dir.parent.name == "backtest" else None
    if prediction_hash and run_log_dir is not None:
        candidate = run_log_dir / "predictions" / prediction_hash / "predictions.parquet"
        if candidate.exists():
            return candidate

    registry_path = run_log_dir / "registry.db" if run_log_dir is not None else None
    if registry_path is not None and registry_path.exists():
        try:
            with sqlite3.connect(registry_path) as conn:
                row = conn.execute(
                    "SELECT prediction_hash FROM backtest_runs WHERE backtest_hash = ?",
                    (backtest_dir.name,),
                ).fetchone()
            if row and row[0]:
                candidate = run_log_dir / "predictions" / row[0] / "predictions.parquet"
                if candidate.exists():
                    return candidate
        except sqlite3.Error as exc:
            import warnings

            warnings.warn(
                f"Registry lookup for prediction_hash failed: {exc}. "
                "Predictions may be unavailable.",
                stacklevel=2,
            )
            return None
    return None


def _trades_from_dataframe(trades_df: pl.DataFrame) -> list[Trade]:
    trades: list[Trade] = []
    for row in trades_df.to_dicts():
        trades.append(
            Trade(
                symbol=str(row["symbol"]),
                entry_time=_to_datetime(row["entry_time"]),
                exit_time=_to_datetime(row["exit_time"]),
                entry_price=float(row["entry_price"]),
                exit_price=float(row["exit_price"]),
                quantity=float(row["quantity"]),
                pnl=float(row["pnl"]),
                pnl_percent=float(row["pnl_percent"]),
                bars_held=int(row["bars_held"]),
                fees=float(row.get("fees", 0.0) or 0.0),
                exit_slippage=float(row.get("exit_slippage", 0.0) or 0.0),
                exit_reason=str(row.get("exit_reason", "signal") or "signal"),
                status=str(row.get("status", "closed") or "closed"),
                mfe=float(row.get("mfe", 0.0) or 0.0),
                mae=float(row.get("mae", 0.0) or 0.0),
                entry_slippage=float(row.get("entry_slippage", 0.0) or 0.0),
                multiplier=float(row.get("multiplier", 1.0) or 1.0),
                entry_quote_mid_price=_optional_float(row.get("entry_quote_mid_price")),
                entry_bid_price=_optional_float(row.get("entry_bid_price")),
                entry_ask_price=_optional_float(row.get("entry_ask_price")),
                entry_spread=_optional_float(row.get("entry_spread")),
                entry_available_size=_optional_float(row.get("entry_available_size")),
                exit_quote_mid_price=_optional_float(row.get("exit_quote_mid_price")),
                exit_bid_price=_optional_float(row.get("exit_bid_price")),
                exit_ask_price=_optional_float(row.get("exit_ask_price")),
                exit_spread=_optional_float(row.get("exit_spread")),
                exit_available_size=_optional_float(row.get("exit_available_size")),
            )
        )
    return trades


def _equity_curve_from_parquet(path: Path) -> list[tuple[datetime, float]]:
    """Load equity curve from a real equity.parquet artifact."""
    df = pl.read_parquet(path)
    ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    eq_col = "equity" if "equity" in df.columns else df.columns[1]
    return [
        (_to_datetime(row[ts_col]), float(row[eq_col]))
        for row in df.select([ts_col, eq_col]).iter_rows(named=True)
    ]


def _portfolio_state_from_parquet(
    path: Path,
) -> list[tuple[datetime, float, float, float, float, int]]:
    """Load portfolio state from a real portfolio_state.parquet artifact."""
    df = pl.read_parquet(path)
    rows: list[tuple[datetime, float, float, float, float, int]] = []
    for row in df.iter_rows(named=True):
        rows.append((
            _to_datetime(row["timestamp"]),
            float(row.get("equity", 0.0) or 0.0),
            float(row.get("cash", 0.0) or 0.0),
            float(row.get("gross_exposure", 0.0) or 0.0),
            float(row.get("net_exposure", 0.0) or 0.0),
            int(row.get("open_positions", 0) or 0),
        ))
    return rows


def _fills_from_parquet(path: Path) -> list[Fill]:
    """Load fills from a real fills.parquet artifact."""
    df = pl.read_parquet(path)
    fills: list[Fill] = []
    for row in df.iter_rows(named=True):
        side_val = row.get("side", "buy")
        side = OrderSide.BUY if side_val == "buy" else OrderSide.SELL
        fills.append(
            Fill(
                order_id=str(row.get("order_id", "")),
                asset=str(row.get("asset", row.get("symbol", ""))),
                side=side,
                quantity=float(row.get("quantity", 0.0) or 0.0),
                price=float(row.get("price", 0.0) or 0.0),
                timestamp=_to_datetime(row["timestamp"]),
                rebalance_id=row.get("rebalance_id"),
                commission=float(row.get("commission", 0.0) or 0.0),
                slippage=float(row.get("slippage", 0.0) or 0.0),
                order_type=str(row.get("order_type", "") or ""),
                limit_price=_optional_float(row.get("limit_price")),
                stop_price=_optional_float(row.get("stop_price")),
                price_source=str(row.get("price_source", "") or ""),
                reference_price=_optional_float(row.get("reference_price")),
                quote_mid_price=_optional_float(row.get("quote_mid_price")),
                bid_price=_optional_float(row.get("bid_price")),
                ask_price=_optional_float(row.get("ask_price")),
                spread=_optional_float(row.get("spread")),
                bid_size=_optional_float(row.get("bid_size")),
                ask_size=_optional_float(row.get("ask_size")),
                available_size=_optional_float(row.get("available_size")),
            )
        )
    return fills


def _equity_curve_from_daily_returns(
    daily_returns_df: pl.DataFrame,
    spec: dict[str, Any],
) -> list[tuple[datetime, float]]:
    if daily_returns_df.is_empty():
        return []
    timestamp_col = "timestamp" if "timestamp" in daily_returns_df.columns else daily_returns_df.columns[0]
    return_col = "daily_return" if "daily_return" in daily_returns_df.columns else daily_returns_df.columns[1]
    initial_cash = (
        spec.get("backtest_config", {})
        .get("cash", {})
        .get("initial", 100_000.0)
    )
    equity = float(initial_cash)
    curve: list[tuple[datetime, float]] = []
    for row in daily_returns_df.select([timestamp_col, return_col]).iter_rows(named=True):
        equity *= 1.0 + float(row[return_col] or 0.0)
        curve.append((_to_datetime(row[timestamp_col]), equity))
    return curve


def _load_weights_as_signal_surface_from_dataframe(weights_df: pl.DataFrame) -> pl.DataFrame:
    if weights_df.is_empty():
        return pl.DataFrame()
    rename_map: dict[str, str] = {}
    if "symbol" in weights_df.columns and "asset" not in weights_df.columns:
        rename_map["symbol"] = "asset"
    if "weight" in weights_df.columns:
        rename_map["weight"] = "signal_value"
    if rename_map:
        weights_df = weights_df.rename(rename_map)
    if "selected" not in weights_df.columns and "signal_value" in weights_df.columns:
        weights_df = weights_df.with_columns((pl.col("signal_value").abs() > 1e-9).alias("selected"))
    return weights_df


def _normalize_surface_from_feed_spec(
    surface: pl.DataFrame,
    feed_spec: Any | None,
) -> pl.DataFrame:
    if surface.is_empty() or feed_spec is None:
        return surface

    rename_map: dict[str, str] = {}
    timestamp_col = getattr(feed_spec, "timestamp_col", None)
    if (
        isinstance(timestamp_col, str)
        and timestamp_col
        and timestamp_col != "timestamp"
        and timestamp_col in surface.columns
        and "timestamp" not in surface.columns
    ):
        rename_map[timestamp_col] = "timestamp"

    entity_col = _feed_entity_col(feed_spec)
    if (
        entity_col is not None
        and entity_col != "asset"
        and entity_col in surface.columns
        and "asset" not in surface.columns
    ):
        rename_map[entity_col] = "asset"

    return surface.rename(rename_map) if rename_map else surface


def _feed_entity_col(feed_spec: Any | None) -> str | None:
    if feed_spec is None:
        return None
    entity_col = getattr(feed_spec, "entity_col", None)
    if entity_col is None:
        return None
    if isinstance(entity_col, str):
        return entity_col
    entity_values = [str(value) for value in entity_col]
    if len(entity_values) != 1:
        return None
    return entity_values[0]


def _load_weights_as_signal_surface(weights_path: Path) -> pl.DataFrame:
    return _load_weights_as_signal_surface_from_dataframe(_load_weights_dataframe(weights_path))


def _infer_active_start_timestamp(
    *,
    daily_returns_df: pl.DataFrame,
    trades_df: pl.DataFrame,
    weights_df: pl.DataFrame,
) -> datetime | None:
    candidates: list[datetime] = []

    returns_ts_col = _first_present_column(daily_returns_df, ("timestamp", "date"))
    returns_col = _first_present_column(daily_returns_df, ("daily_return", "return", "return_pct"))
    if returns_ts_col is not None and returns_col is not None and not daily_returns_df.is_empty():
        active_returns = daily_returns_df.filter(pl.col(returns_col).abs() > 1e-12)
        if not active_returns.is_empty():
            candidates.append(_to_datetime(active_returns[returns_ts_col].min()))

    if not trades_df.is_empty():
        trade_ts_col = _first_present_column(trades_df, ("entry_time", "timestamp"))
        if trade_ts_col is not None:
            candidates.append(_to_datetime(trades_df[trade_ts_col].min()))

    if not weights_df.is_empty():
        weight_ts_col = _first_present_column(weights_df, ("timestamp", "date"))
        weight_col = _first_present_column(weights_df, ("weight", "signal_value"))
        if weight_ts_col is not None and weight_col is not None:
            active_weights = weights_df.filter(pl.col(weight_col).abs() > 1e-9)
            if not active_weights.is_empty():
                candidates.append(_to_datetime(active_weights[weight_ts_col].min()))

    return min(candidates) if candidates else None


def _trim_dataframe_from_timestamp(
    df: pl.DataFrame,
    *,
    timestamp_col: str | None,
    start: datetime,
) -> pl.DataFrame:
    if df.is_empty() or timestamp_col is None:
        return df
    if isinstance(df[timestamp_col].dtype, pl.String):
        df = df.with_columns(
            pl.coalesce(
                [
                    pl.col(timestamp_col).str.to_datetime(strict=False),
                    pl.col(timestamp_col).str.to_date(strict=False).cast(pl.Datetime),
                ]
            ).alias(timestamp_col)
        )
    dtype = df[timestamp_col].dtype
    if isinstance(dtype, pl.String):
        return df
    literal = _align_literal_for_series(start, dtype)
    return df.filter(pl.col(timestamp_col) >= pl.lit(literal, dtype=dtype))


def _align_literal_for_series(timestamp: datetime, dtype: pl.DataType) -> datetime | date:
    if isinstance(dtype, pl.Date):
        return timestamp.date()
    return timestamp.replace(tzinfo=None)


def _first_present_column(df: pl.DataFrame, candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _align_timestamp_precision(*dfs: pl.DataFrame) -> tuple[pl.DataFrame, ...]:
    """Cast all DataFrames' timestamp column to microsecond precision.

    Polars join requires matching dtypes; Parquet files from different
    sources may use datetime[ms] vs datetime[μs].
    """
    out = []
    target = pl.Datetime("us")
    for df in dfs:
        if "timestamp" in df.columns and df["timestamp"].dtype != target:
            df = df.with_columns(pl.col("timestamp").cast(target))
        out.append(df)
    return tuple(out)


def _portfolio_state_from_weights(
    weights_df: pl.DataFrame,
    equity_curve: list[tuple[datetime, float]],
) -> list[tuple[datetime, float, float, float, float, int]]:
    if weights_df.is_empty() or not equity_curve:
        return []

    weights = weights_df.rename(
        {
            "symbol": "asset",
            "weight": "signal_value",
        }
    ) if "symbol" in weights_df.columns else weights_df.rename({"weight": "signal_value"})
    summary = (
        weights.group_by("timestamp")
        .agg(
            [
                pl.col("signal_value").abs().sum().alias("gross_weight"),
                pl.col("signal_value").sum().alias("net_weight"),
                (pl.col("signal_value").abs() > 1e-9)
                .sum()
                .cast(pl.Int32)
                .alias("open_positions"),
            ]
        )
        .sort("timestamp")
    )
    equity_df = pl.DataFrame(
        {
            "timestamp": [timestamp for timestamp, _ in equity_curve],
            "equity": [float(value) for _, value in equity_curve],
        }
    )
    equity_df, summary = _align_timestamp_precision(equity_df, summary)
    timeline = (
        equity_df.join(summary, on="timestamp", how="left")
        .with_columns(
            [
                pl.col("gross_weight").fill_null(0.0),
                pl.col("net_weight").fill_null(0.0),
                pl.col("open_positions").fill_null(0).cast(pl.Int32),
            ]
        )
        .with_columns(
            [
                (pl.col("equity") * pl.col("gross_weight")).alias("gross_exposure"),
                (pl.col("equity") * pl.col("net_weight")).alias("net_exposure"),
                (pl.col("equity") * (1.0 - pl.col("gross_weight"))).alias("cash"),
            ]
        )
        .select(
            [
                "timestamp",
                "equity",
                "cash",
                "gross_exposure",
                "net_exposure",
                "open_positions",
            ]
        )
    )
    return [tuple(row) for row in timeline.iter_rows()]


def _fills_from_weights(
    weights_df: pl.DataFrame,
    equity_curve: list[tuple[datetime, float]],
) -> list[Fill]:
    if weights_df.is_empty() or not equity_curve or "symbol" not in weights_df.columns:
        return []

    weights = (
        weights_df.sort(["symbol", "timestamp"])
        .with_columns(
            (
                pl.col("weight") - pl.col("weight").shift(1).fill_null(0.0).over("symbol")
            ).alias("weight_delta")
        )
        .filter(pl.col("weight_delta").abs() > 1e-6)
    )
    if weights.is_empty():
        return []

    equity_df = pl.DataFrame(
        {
            "timestamp": [timestamp for timestamp, _ in equity_curve],
            "equity": [float(value) for _, value in equity_curve],
        }
    )
    weights, equity_df = _align_timestamp_precision(weights, equity_df)
    enriched = weights.join(equity_df, on="timestamp", how="left").with_columns(
        pl.col("equity").fill_null(strategy="forward").fill_null(strategy="backward")
    )

    fills: list[Fill] = []
    for index, row in enumerate(enriched.iter_rows(named=True)):
        delta = float(row["weight_delta"])
        timestamp = _to_datetime(row["timestamp"])
        fills.append(
            Fill(
                order_id=f"weights-{index}",
                rebalance_id=timestamp.isoformat(),
                asset=str(row["symbol"]),
                side=OrderSide.BUY if delta > 0 else OrderSide.SELL,
                quantity=abs(delta) * float(row["equity"] or 0.0),
                price=1.0,
                timestamp=timestamp,
                commission=0.0,
                slippage=0.0,
            )
        )
    return fills


def _build_strategy_metadata(
    spec: dict[str, Any],
    predictions_df: pl.DataFrame | None,
    backtest_dir: Path,
) -> dict[str, Any]:
    strategy_spec = spec.get("strategy", {})
    signal_spec = strategy_spec.get("signal", {})
    allocation_spec = strategy_spec.get("allocation", {})
    config_name = None
    if predictions_df is not None and "config_name" in predictions_df.columns and predictions_df.height > 0:
        config_name = predictions_df["config_name"][0]

    metadata = {
        "strategy_name": config_name or spec.get("preset_id") or backtest_dir.name,
        "strategy_type": "ml",
        "is_ml_strategy": True,
        "mapping_name": signal_spec.get("method"),
        "signal_type": "weight",
        "prediction_type": "score",
    }
    if allocation_spec.get("method"):
        metadata["allocation_method"] = allocation_spec["method"]
    if signal_spec.get("top_k") is not None:
        metadata["selection_description"] = f"top_k={signal_spec['top_k']}"
    return {key: value for key, value in metadata.items() if value is not None}


def _build_report_metadata_from_artifacts(
    backtest_dir: Path,
    strategy_metadata: dict[str, Any],
) -> BacktestReportMetadata:
    strategy_name = str(strategy_metadata.get("strategy_name") or backtest_dir.name)
    subtitle_parts = []
    if strategy_metadata.get("mapping_name"):
        subtitle_parts.append(str(strategy_metadata["mapping_name"]))
    if strategy_metadata.get("allocation_method"):
        subtitle_parts.append(str(strategy_metadata["allocation_method"]))
    return BacktestReportMetadata(
        strategy_name=strategy_name,
        subtitle=" | ".join(subtitle_parts),
        run_id=backtest_dir.name,
    )


def _to_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time())
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value)
        return parsed if isinstance(parsed, datetime) else datetime.combine(parsed, time())
    raise TypeError(f"Cannot coerce {value!r} to datetime")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


__all__ = [
    "BacktestProfile",
    "analyze_backtest_result",
    "compute_metrics_from_result",
    "generate_tearsheet_from_result",
    "generate_tearsheet_from_run_artifacts",
    "portfolio_analysis_from_result",
    "profile_from_run_artifacts",
]
