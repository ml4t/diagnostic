"""Compositional backtest profile built from raw reporting surfaces."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import polars as pl

from ml4t.diagnostic.integration.backtest_analytics import (
    compute_activity_metrics,
    compute_attribution_metrics,
    compute_drawdown_anatomy,
    compute_edge_metrics,
    compute_occupancy_metrics,
    compute_performance_metrics,
)

if TYPE_CHECKING:
    from ml4t.backtest import BacktestResult
else:
    try:
        from ml4t.backtest import BacktestResult
        from ml4t.backtest.analytics.annualization import get_annualization_factor
        from ml4t.backtest.result import enrich_trades_with_signals
    except ImportError as exc:
        raise ImportError(
            "ml4t-backtest integration requires the optional 'ml4t-backtest' package. "
            "Install with: pip install 'ml4t-diagnostic[backtest]'"
        ) from exc


class AvailabilityState(str, Enum):
    """Availability state for a surface, family, or metric."""

    AVAILABLE = "available"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"


@dataclass(frozen=True)
class AvailabilityInfo:
    """Availability metadata for a reporting component."""

    status: AvailabilityState
    reason: str | None = None
    coverage: float | None = None
    fallback: str | None = None


@dataclass(frozen=True)
class BacktestAvailability:
    """Availability metadata for raw surfaces and analytics families."""

    surfaces: dict[str, AvailabilityInfo]
    families: dict[str, AvailabilityInfo]
    metrics: dict[str, AvailabilityInfo]


@dataclass
class BacktestProfile:
    """Lazy analytical profile over a backtest result."""

    result: BacktestResult
    calendar: str | None = None
    benchmark: Any = None
    confidence_intervals: bool = False
    quote_coverage_threshold: float = 0.8
    predictions_override: Any = None
    signals_override: Any = None
    strategy_metadata_override: dict[str, Any] | None = None
    data_sources: dict[str, str] = field(default_factory=dict)
    _performance_cache: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _edge_cache: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _activity_cache: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _occupancy_cache: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _attribution_cache: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _drawdown_cache: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _ml_cache: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _prediction_trades_cache: pl.DataFrame | None = field(default=None, init=False, repr=False)
    _availability_cache: BacktestAvailability | None = field(default=None, init=False, repr=False)
    _summary_cache: dict[str, Any] | None = field(default=None, init=False, repr=False)

    @property
    def resolved_calendar(self) -> str | None:
        """Return explicit calendar override or the result calendar."""
        return self.calendar or (
            self.result.config.resolved_calendar if self.result.config else None
        )

    @property
    def periods_per_year(self) -> int:
        """Return annualization factor for the resolved calendar."""
        return get_annualization_factor(self.resolved_calendar)

    @property
    def trades_df(self) -> pl.DataFrame:
        """Return the raw trade surface."""
        return self.result.to_trades_dataframe()

    @property
    def fills_df(self) -> pl.DataFrame:
        """Return the raw fill surface."""
        return self.result.to_fills_dataframe()

    @property
    def portfolio_state_df(self) -> pl.DataFrame:
        """Return the raw portfolio-state surface."""
        return self.result.to_portfolio_state_dataframe()

    @property
    def equity_df(self) -> pl.DataFrame:
        """Return the raw equity surface."""
        return self.result.to_equity_dataframe()

    @property
    def predictions_df(self) -> pl.DataFrame:
        """Return the optional prediction surface if the result exposes one."""
        if self.predictions_override is not None:
            surface = _normalize_prediction_surface(
                _normalize_optional_surface(_coerce_optional_surface(self.predictions_override))
            )
            if not surface.is_empty():
                return surface
        return _extract_optional_surface(
            self.result,
            method_names=("to_predictions_df", "to_predictions_dataframe"),
            attr_names=("predictions_df", "predictions"),
            normalizer=_normalize_prediction_surface,
        )

    @property
    def signals_df(self) -> pl.DataFrame:
        """Return the optional mapped-signal surface if the result exposes one."""
        if self.signals_override is not None:
            surface = _normalize_signal_surface(
                _normalize_optional_surface(_coerce_optional_surface(self.signals_override))
            )
            if not surface.is_empty():
                return surface
        return _extract_optional_surface(
            self.result,
            method_names=("to_signals_df", "to_signals_dataframe"),
            attr_names=("signals_df", "signals"),
            normalizer=_normalize_signal_surface,
        )

    @property
    def has_predictions(self) -> bool:
        """Return whether a prediction surface is available."""
        return not self.predictions_df.is_empty()

    @property
    def has_signals(self) -> bool:
        """Return whether a mapped-signal surface is available."""
        return not self.signals_df.is_empty()

    @property
    def strategy_metadata(self) -> dict[str, Any]:
        """Return normalized strategy metadata when available."""
        if self.strategy_metadata_override:
            return dict(self.strategy_metadata_override)
        metadata = getattr(self.result, "strategy_metadata", None)
        if metadata is None:
            return {}
        if isinstance(metadata, dict):
            return dict(metadata)
        if hasattr(metadata, "model_dump"):
            try:
                dumped = metadata.model_dump()
                return dumped if isinstance(dumped, dict) else {}
            except Exception as exc:
                import warnings

                warnings.warn(
                    f"Strategy metadata model_dump() failed: {exc}",
                    stacklevel=2,
                )
                return {}
        if hasattr(metadata, "to_dict"):
            try:
                dumped = metadata.to_dict()
                return dumped if isinstance(dumped, dict) else {}
            except Exception as exc:
                import warnings

                warnings.warn(
                    f"Strategy metadata to_dict() failed: {exc}",
                    stacklevel=2,
                )
                return {}
        if hasattr(metadata, "__dict__"):
            return {key: value for key, value in vars(metadata).items() if not key.startswith("_")}
        return {}

    @property
    def prediction_enriched_trades_df(self) -> pl.DataFrame:
        """Return trades enriched with prediction columns when possible."""
        if self._prediction_trades_cache is None:
            predictions_df = self.predictions_df
            trades_df = self.trades_df
            if predictions_df.is_empty() or trades_df.is_empty():
                self._prediction_trades_cache = pl.DataFrame()
            else:
                timestamp_col = _find_first_available_column(
                    predictions_df,
                    ("timestamp", "date", "session_date"),
                )
                if timestamp_col is None:
                    self._prediction_trades_cache = pl.DataFrame()
                else:
                    asset_col = "asset" if "asset" in predictions_df.columns else None
                    try:
                        self._prediction_trades_cache = enrich_trades_with_signals(
                            trades_df=trades_df,
                            signals_df=predictions_df,
                            timestamp_col=timestamp_col,
                            asset_col=asset_col,
                        )
                    except Exception as exc:
                        import warnings

                        warnings.warn(
                            f"Prediction enrichment failed: {exc}. "
                            "ML trade-alignment charts will be unavailable.",
                            stacklevel=2,
                        )
                        self._prediction_trades_cache = pl.DataFrame()
        return self._prediction_trades_cache

    @property
    def daily_returns(self) -> pl.Series:
        """Return the session-aware daily return surface."""
        return self.result.to_daily_returns(calendar=self.resolved_calendar)

    @property
    def performance(self) -> dict[str, Any]:
        """Return performance and stability analytics."""
        if self._performance_cache is None:
            self._performance_cache = compute_performance_metrics(
                daily_returns=self.daily_returns,
                equity_df=self.equity_df,
                periods_per_year=self.periods_per_year,
                confidence_intervals=self.confidence_intervals,
            )
        return self._performance_cache

    @property
    def edge(self) -> dict[str, Any]:
        """Return trade-lifecycle analytics."""
        if self._edge_cache is None:
            self._edge_cache = compute_edge_metrics(self.trades_df)
        return self._edge_cache

    @property
    def activity(self) -> dict[str, Any]:
        """Return fill and rebalance activity analytics."""
        if self._activity_cache is None:
            self._activity_cache = compute_activity_metrics(self.fills_df, self.portfolio_state_df)
        return self._activity_cache

    @property
    def occupancy(self) -> dict[str, Any]:
        """Return occupancy analytics from portfolio-state snapshots."""
        if self._occupancy_cache is None:
            self._occupancy_cache = compute_occupancy_metrics(self.portfolio_state_df)
        return self._occupancy_cache

    @property
    def attribution(self) -> dict[str, Any]:
        """Return symbol-level contribution and burden analytics."""
        if self._attribution_cache is None:
            self._attribution_cache = compute_attribution_metrics(self.trades_df, self.fills_df)
        return self._attribution_cache

    @property
    def drawdown(self) -> dict[str, Any]:
        """Return drawdown episodes and peak-to-trough contributors."""
        if self._drawdown_cache is None:
            self._drawdown_cache = compute_drawdown_anatomy(self.equity_df, self.trades_df)
        return self._drawdown_cache

    @property
    def ml(self) -> dict[str, Any]:
        """Return lightweight prediction-surface summary metrics when available."""
        if self._ml_cache is None:
            predictions_df = self.predictions_df
            signals_df = self.signals_df
            strategy_metadata = self.strategy_metadata
            if predictions_df.is_empty() and signals_df.is_empty() and not strategy_metadata:
                self._ml_cache = {
                    "metrics": {},
                    "available": False,
                    "has_predictions": False,
                    "has_signals": False,
                    "strategy_metadata": {},
                }
            else:
                prediction_trades_df = self.prediction_enriched_trades_df
                metrics: dict[str, Any] = {
                    "n_predictions": predictions_df.height,
                    "n_prediction_columns": len(predictions_df.columns),
                    "n_signals": signals_df.height,
                    "n_signal_columns": len(signals_df.columns),
                    "translation_ready": not predictions_df.is_empty()
                    and not signals_df.is_empty(),
                }
                if "asset" in predictions_df.columns:
                    metrics["n_prediction_assets"] = predictions_df["asset"].n_unique()
                if "asset" in signals_df.columns:
                    metrics["n_signal_assets"] = signals_df["asset"].n_unique()
                selected_col = (
                    "selected"
                    if "selected" in signals_df.columns
                    else "selected"
                    if "selected" in predictions_df.columns
                    else None
                )
                selected_df = signals_df if "selected" in signals_df.columns else predictions_df
                if selected_col is not None:
                    selected = selected_df[selected_col].cast(pl.Int8(), strict=False)
                    metrics["selection_rate"] = float(selected.mean()) if selected.len() else None
                trade_link_df = signals_df if "trade_id" in signals_df.columns else predictions_df
                if "trade_id" in trade_link_df.columns:
                    linked = trade_link_df["trade_id"].is_not_null()
                    metrics["trade_link_rate"] = float(linked.mean()) if linked.len() else None
                entry_prediction_columns = [
                    column
                    for column in prediction_trades_df.columns
                    if column.startswith("entry_")
                    and prediction_trades_df.schema[column].is_numeric()
                ]
                if entry_prediction_columns:
                    metrics["entry_prediction_columns"] = entry_prediction_columns
                    coverage = prediction_trades_df[entry_prediction_columns[0]].is_not_null()
                    metrics["trade_prediction_coverage"] = (
                        float(coverage.mean()) if coverage.len() else None
                    )
                self._ml_cache = {
                    "metrics": metrics,
                    "available": True,
                    "has_predictions": not predictions_df.is_empty(),
                    "has_signals": not signals_df.is_empty(),
                    "strategy_metadata": strategy_metadata,
                }
        return self._ml_cache

    @property
    def availability(self) -> BacktestAvailability:
        """Return explicit availability metadata for surfaces and families."""
        if self._availability_cache is None:
            self._availability_cache = _build_availability(
                trades_df=self.trades_df,
                fills_df=self.fills_df,
                portfolio_state_df=self.portfolio_state_df,
                equity_df=self.equity_df,
                predictions_df=self.predictions_df,
                signals_df=self.signals_df,
                quote_coverage_threshold=self.quote_coverage_threshold,
            )
        return self._availability_cache

    @property
    def summary(self) -> dict[str, Any]:
        """Return a flat summary for legacy bridge compatibility."""
        if self._summary_cache is None:
            summary: dict[str, Any] = {}
            summary.update(self.performance["metrics"])
            summary.update(self.edge["metrics"])
            summary.update(self.activity["metrics"])
            summary.update(self.occupancy["metrics"])
            summary.update(self.drawdown["metrics"])
            self._summary_cache = summary
        return self._summary_cache


def _build_availability(
    trades_df: pl.DataFrame,
    fills_df: pl.DataFrame,
    portfolio_state_df: pl.DataFrame,
    equity_df: pl.DataFrame,
    predictions_df: pl.DataFrame,
    signals_df: pl.DataFrame,
    quote_coverage_threshold: float,
) -> BacktestAvailability:
    surfaces = {
        "trades": AvailabilityInfo(
            AvailabilityState.AVAILABLE
            if not trades_df.is_empty()
            else AvailabilityState.UNAVAILABLE,
            None if not trades_df.is_empty() else "No trade rows were emitted.",
        ),
        "fills": AvailabilityInfo(
            AvailabilityState.AVAILABLE
            if not fills_df.is_empty()
            else AvailabilityState.UNAVAILABLE,
            None if not fills_df.is_empty() else "No fill rows were emitted.",
        ),
        "portfolio_state": AvailabilityInfo(
            (
                AvailabilityState.AVAILABLE
                if not portfolio_state_df.is_empty()
                else AvailabilityState.UNAVAILABLE
            ),
            None if not portfolio_state_df.is_empty() else "No portfolio state rows were emitted.",
        ),
        "equity": AvailabilityInfo(
            AvailabilityState.AVAILABLE
            if not equity_df.is_empty()
            else AvailabilityState.UNAVAILABLE,
            None if not equity_df.is_empty() else "No equity curve rows were emitted.",
        ),
        "predictions": AvailabilityInfo(
            (
                AvailabilityState.AVAILABLE
                if not predictions_df.is_empty()
                else AvailabilityState.UNAVAILABLE
            ),
            None if not predictions_df.is_empty() else "No prediction surface was attached.",
        ),
        "signals": AvailabilityInfo(
            (
                AvailabilityState.AVAILABLE
                if not signals_df.is_empty()
                else AvailabilityState.UNAVAILABLE
            ),
            None if not signals_df.is_empty() else "No mapped signal surface was attached.",
        ),
    }

    quote_coverage = 0.0
    if not fills_df.is_empty():
        quote_rows = fills_df.select(
            pl.any_horizontal(
                [
                    pl.col("quote_mid_price").is_not_null(),
                    pl.col("bid_price").is_not_null(),
                    pl.col("ask_price").is_not_null(),
                ]
            ).sum()
        ).item()
        quote_coverage = float(quote_rows) / float(fills_df.height)

    execution_status = AvailabilityState.UNAVAILABLE
    execution_reason = "Quote-aware execution fields are missing."
    if not fills_df.is_empty():
        if quote_coverage >= quote_coverage_threshold:
            execution_status = AvailabilityState.AVAILABLE
            execution_reason = None
        elif quote_coverage > 0:
            execution_status = AvailabilityState.PARTIAL
            execution_reason = "Quote-aware coverage is below the rendering threshold."

    activity_status = AvailabilityState.UNAVAILABLE
    activity_reason = "No fill rows were emitted."
    activity_fallback: str | None = None
    if not fills_df.is_empty():
        if portfolio_state_df.is_empty():
            activity_status = AvailabilityState.DEGRADED
            activity_reason = "Turnover and cost-drag series require portfolio_state."
            activity_fallback = "Use fill counts, notional, and rebalance summaries only."
        else:
            activity_status = AvailabilityState.AVAILABLE
            activity_reason = None

    attribution_status = AvailabilityState.UNAVAILABLE
    attribution_reason = "Neither trades nor fills are available."
    attribution_fallback: str | None = None
    if not trades_df.is_empty() and not fills_df.is_empty():
        attribution_status = AvailabilityState.AVAILABLE
        attribution_reason = None
    elif not trades_df.is_empty() or not fills_df.is_empty():
        attribution_status = AvailabilityState.DEGRADED
        attribution_reason = "Only one raw surface is available for attribution."
        attribution_fallback = "Partial symbol contribution views are still usable."

    ml_status = AvailabilityState.UNAVAILABLE
    ml_reason = "ML diagnostics require predictions or signals."
    ml_fallback: str | None = None
    if not predictions_df.is_empty() and not signals_df.is_empty():
        ml_status = AvailabilityState.AVAILABLE
        ml_reason = None
    elif not predictions_df.is_empty() or not signals_df.is_empty():
        ml_status = AvailabilityState.PARTIAL
        ml_reason = "Only one ML surface is available; translation analysis is incomplete."
        ml_fallback = "Use the available surface for raw model or decision summaries only."

    translation_status = AvailabilityState.UNAVAILABLE
    translation_reason = "Prediction translation requires both predictions and signals."
    translation_fallback: str | None = None
    if not predictions_df.is_empty() and not signals_df.is_empty():
        translation_status = AvailabilityState.AVAILABLE
        translation_reason = None
    elif not predictions_df.is_empty() or not signals_df.is_empty():
        translation_status = AvailabilityState.DEGRADED
        translation_reason = "Only one of predictions/signals is available."
        translation_fallback = (
            "Translation analysis becomes available once both surfaces are attached."
        )

    families = {
        "performance": AvailabilityInfo(
            AvailabilityState.AVAILABLE
            if not equity_df.is_empty()
            else AvailabilityState.UNAVAILABLE,
            None if not equity_df.is_empty() else "Performance requires an equity surface.",
        ),
        "edge": AvailabilityInfo(
            AvailabilityState.AVAILABLE
            if not trades_df.is_empty()
            else AvailabilityState.UNAVAILABLE,
            None if not trades_df.is_empty() else "Edge metrics require trade rows.",
        ),
        "activity": AvailabilityInfo(
            activity_status,
            activity_reason,
            fallback=activity_fallback,
        ),
        "occupancy": AvailabilityInfo(
            (
                AvailabilityState.AVAILABLE
                if not portfolio_state_df.is_empty()
                else AvailabilityState.UNAVAILABLE
            ),
            None if not portfolio_state_df.is_empty() else "Occupancy requires portfolio_state.",
        ),
        "attribution": AvailabilityInfo(
            attribution_status,
            attribution_reason,
            fallback=attribution_fallback,
        ),
        "execution": AvailabilityInfo(
            execution_status,
            execution_reason,
            coverage=quote_coverage if not fills_df.is_empty() else None,
            fallback=(
                "Use commissions and slippage from fills only."
                if not fills_df.is_empty() and execution_status != AvailabilityState.AVAILABLE
                else None
            ),
        ),
        "drawdown": AvailabilityInfo(
            AvailabilityState.AVAILABLE
            if not equity_df.is_empty()
            else AvailabilityState.UNAVAILABLE,
            None if not equity_df.is_empty() else "Drawdown anatomy requires an equity surface.",
        ),
        "ml": AvailabilityInfo(
            ml_status,
            ml_reason,
            fallback=ml_fallback,
        ),
    }

    metrics = {
        "turnover": AvailabilityInfo(
            (
                AvailabilityState.AVAILABLE
                if not fills_df.is_empty() and not portfolio_state_df.is_empty()
                else AvailabilityState.UNAVAILABLE
            ),
            (
                None
                if not fills_df.is_empty() and not portfolio_state_df.is_empty()
                else "Turnover requires both fills and portfolio_state."
            ),
        ),
        "cost_drag": AvailabilityInfo(
            (
                AvailabilityState.AVAILABLE
                if not fills_df.is_empty() and not portfolio_state_df.is_empty()
                else AvailabilityState.DEGRADED
                if not fills_df.is_empty()
                else AvailabilityState.UNAVAILABLE
            ),
            (
                None
                if not fills_df.is_empty() and not portfolio_state_df.is_empty()
                else "Cost drag over time requires portfolio_state."
                if not fills_df.is_empty()
                else "Cost drag requires fill rows."
            ),
            fallback="Use total implementation cost only." if not fills_df.is_empty() else None,
        ),
        "execution_audit": AvailabilityInfo(
            execution_status,
            execution_reason,
            coverage=quote_coverage if not fills_df.is_empty() else None,
        ),
        "prediction_translation": AvailabilityInfo(
            translation_status,
            translation_reason,
            fallback=translation_fallback,
        ),
    }
    return BacktestAvailability(surfaces=surfaces, families=families, metrics=metrics)


def _extract_optional_surface(
    result: BacktestResult,
    *,
    method_names: tuple[str, ...],
    attr_names: tuple[str, ...],
    normalizer: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
) -> pl.DataFrame:
    for method_name in method_names:
        surface_method = getattr(result, method_name, None)
        if callable(surface_method):
            surface = _normalize_optional_surface(_coerce_optional_surface(surface_method()))
            if normalizer is not None:
                surface = normalizer(surface)
            if not surface.is_empty():
                return surface

    for attr_name in attr_names:
        surface = _normalize_optional_surface(
            _coerce_optional_surface(getattr(result, attr_name, None))
        )
        if normalizer is not None:
            surface = normalizer(surface)
        if not surface.is_empty():
            return surface

    return pl.DataFrame()


def _coerce_optional_surface(surface: Any) -> pl.DataFrame:
    if surface is None:
        return pl.DataFrame()
    if isinstance(surface, pl.DataFrame):
        return surface
    if hasattr(surface, "to_pandas"):
        try:
            return pl.from_pandas(surface)  # pragma: no cover - optional pandas path
        except Exception as exc:
            import warnings

            warnings.warn(
                f"Failed to convert pandas surface to Polars: {exc}. "
                "Surface will be treated as unavailable.",
                stacklevel=2,
            )
            return pl.DataFrame()
    if isinstance(surface, list):
        try:
            return pl.DataFrame(surface)
        except Exception as exc:
            import warnings

            warnings.warn(
                f"Failed to convert list surface to Polars DataFrame: {exc}. "
                "Surface will be treated as unavailable.",
                stacklevel=2,
            )
            return pl.DataFrame()
    return pl.DataFrame()


def _find_first_available_column(df: pl.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _normalize_optional_surface(surface: pl.DataFrame) -> pl.DataFrame:
    if surface.is_empty():
        return surface

    rename_map: dict[str, str] = {}
    if "asset" not in surface.columns and "symbol" in surface.columns:
        rename_map["symbol"] = "asset"
    if rename_map:
        surface = surface.rename(rename_map)

    timestamp_col = _find_first_available_column(surface, ("timestamp", "date", "session_date"))
    if timestamp_col is None:
        return surface

    dtype = surface.schema.get(timestamp_col)
    if dtype == pl.String:
        parsed_datetime = pl.col(timestamp_col).str.strptime(pl.Datetime, strict=False)
        parsed_date = pl.col(timestamp_col).str.strptime(pl.Date, strict=False).cast(pl.Datetime)
        parsed_expr = pl.coalesce([parsed_datetime, parsed_date])
        preview = surface.select(parsed_expr.alias("_parsed_ts"))
        if preview["_parsed_ts"].is_not_null().any():
            surface = surface.with_columns(parsed_expr.alias(timestamp_col))

    if timestamp_col != "timestamp":
        surface = surface.rename({timestamp_col: "timestamp"})

    return surface


def _normalize_prediction_surface(surface: pl.DataFrame) -> pl.DataFrame:
    if surface.is_empty():
        return surface
    rename_map: dict[str, str] = {}
    if "prediction_value" not in surface.columns:
        if "prediction" in surface.columns:
            rename_map["prediction"] = "prediction_value"
        elif "score" in surface.columns:
            rename_map["score"] = "prediction_value"
        elif "y_score" in surface.columns:
            rename_map["y_score"] = "prediction_value"
        elif "y_pred" in surface.columns:
            rename_map["y_pred"] = "prediction_value"
        elif "ml_score" in surface.columns:
            rename_map["ml_score"] = "prediction_value"
        elif "probability" in surface.columns:
            rename_map["probability"] = "prediction_value"
    return surface.rename(rename_map) if rename_map else surface


def _normalize_signal_surface(surface: pl.DataFrame) -> pl.DataFrame:
    if surface.is_empty():
        return surface
    rename_map: dict[str, str] = {}
    if "signal_value" not in surface.columns:
        if "signal" in surface.columns:
            rename_map["signal"] = "signal_value"
        elif "weight" in surface.columns:
            rename_map["weight"] = "signal_value"
    return surface.rename(rename_map) if rename_map else surface


def analyze_backtest_result(
    result: BacktestResult,
    calendar: str | None = None,
    benchmark: Any = None,
    confidence_intervals: bool = False,
    quote_coverage_threshold: float = 0.8,
) -> BacktestProfile:
    """Build a lazy analytical profile from a BacktestResult."""
    return BacktestProfile(
        result=result,
        calendar=calendar,
        benchmark=benchmark,
        confidence_intervals=confidence_intervals,
        quote_coverage_threshold=quote_coverage_threshold,
    )
