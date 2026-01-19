"""Dashboard data normalization.

Converts various input formats (dict, TradeShapResult) into the unified
DashboardBundle for consumption by all dashboard tabs.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ml4t.diagnostic.evaluation.trade_dashboard.io import coerce_result_to_dict
from ml4t.diagnostic.evaluation.trade_dashboard.types import DashboardBundle, DashboardConfig

if TYPE_CHECKING:
    from ml4t.diagnostic.evaluation.trade_shap.models import TradeShapResult


def normalize_result(
    result: TradeShapResult | dict[str, Any],
    config: DashboardConfig | None = None,
) -> DashboardBundle:
    """Normalize result into a DashboardBundle.

    This is the single point of schema adaptation. All tabs receive the
    normalized DashboardBundle and don't need to handle dict/object branching.

    Parameters
    ----------
    result : TradeShapResult or dict
        Analysis result in either format.
    config : DashboardConfig, optional
        Dashboard configuration.

    Returns
    -------
    DashboardBundle
        Normalized data container with:
        - trades_df sorted chronologically
        - returns array (prefers return_pct, falls back to pnl)
        - normalized explanations and patterns
    """
    if config is None:
        config = DashboardConfig()

    # Convert to dict if needed
    result_dict = coerce_result_to_dict(result)

    # Extract and normalize explanations
    explanations = result_dict.get("explanations", [])
    normalized_explanations = [_normalize_explanation(exp) for exp in explanations]

    # Build trades DataFrame
    trades_df = _build_trades_df(normalized_explanations)

    # Sort chronologically for time-series tests
    if "entry_time" in trades_df.columns and not trades_df["entry_time"].isna().all():
        trades_df = trades_df.sort_values("entry_time", ascending=True).reset_index(drop=True)

    # Extract returns (prefer return_pct, fall back to pnl)
    returns, returns_label = _extract_returns(trades_df)

    # Build patterns DataFrame
    patterns = result_dict.get("error_patterns", [])
    patterns_df = _build_patterns_df(patterns)

    # Extract metadata
    n_analyzed = result_dict.get("n_trades_analyzed", len(explanations))
    n_explained = result_dict.get("n_trades_explained", len(explanations))
    n_failed = result_dict.get("n_trades_failed", 0)
    failed_trades = result_dict.get("failed_trades", [])

    return DashboardBundle(
        trades_df=trades_df,
        returns=returns,
        returns_label=returns_label,
        explanations=normalized_explanations,
        patterns_df=patterns_df,
        n_trades_analyzed=n_analyzed,
        n_trades_explained=n_explained,
        n_trades_failed=n_failed,
        failed_trades=failed_trades,
        config=config,
    )


def _normalize_explanation(exp: dict[str, Any]) -> dict[str, Any]:
    """Normalize a single explanation to stable format.

    Returns dict with stable keys:
    - trade_id: str
    - timestamp: datetime | None
    - shap_vector: list[float]
    - top_features: list[tuple[str, float]]
    - trade_metrics: dict | None
    """
    result: dict[str, Any] = {
        "trade_id": str(exp.get("trade_id", "")),
        "timestamp": _parse_timestamp(exp.get("timestamp")),
        "shap_vector": list(exp.get("shap_vector", [])),
        "top_features": list(exp.get("top_features", [])),
        "trade_metrics": None,
    }

    # Normalize trade_metrics if present
    if exp.get("trade_metrics"):
        tm = exp["trade_metrics"]
        result["trade_metrics"] = {
            "pnl": _safe_float(tm.get("pnl")),
            "return_pct": _safe_float(tm.get("return_pct")),
            "entry_time": _parse_timestamp(tm.get("entry_time")),
            "exit_time": _parse_timestamp(tm.get("exit_time")),
            "duration_days": _safe_float(tm.get("duration_days")),
            "entry_price": _safe_float(tm.get("entry_price")),
            "exit_price": _safe_float(tm.get("exit_price")),
            "symbol": tm.get("symbol"),
        }

    return result


def _build_trades_df(explanations: list[dict[str, Any]]) -> pd.DataFrame:
    """Build trades DataFrame from normalized explanations.

    Returns DataFrame with columns:
    - trade_id: str
    - entry_time: datetime
    - exit_time: datetime (optional)
    - pnl: float
    - return_pct: float (optional)
    - symbol: str (optional)
    - top_feature: str
    - top_shap_value: float
    """
    records = []

    for exp in explanations:
        tm = exp.get("trade_metrics") or {}
        top_features = exp.get("top_features", [])

        record = {
            "trade_id": exp.get("trade_id", ""),
            "entry_time": tm.get("entry_time") or exp.get("timestamp"),
            "exit_time": tm.get("exit_time"),
            "pnl": tm.get("pnl"),
            "return_pct": tm.get("return_pct"),
            "duration_days": tm.get("duration_days"),
            "entry_price": tm.get("entry_price"),
            "exit_price": tm.get("exit_price"),
            "symbol": tm.get("symbol"),
            "top_feature": top_features[0][0] if top_features else None,
            "top_shap_value": top_features[0][1] if top_features else None,
        }
        records.append(record)

    if not records:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(
            columns=[
                "trade_id",
                "entry_time",
                "exit_time",
                "pnl",
                "return_pct",
                "duration_days",
                "entry_price",
                "exit_price",
                "symbol",
                "top_feature",
                "top_shap_value",
            ]
        )

    return pd.DataFrame(records)


def _extract_returns(trades_df: pd.DataFrame) -> tuple[np.ndarray | None, str]:
    """Extract returns array from trades DataFrame.

    Prefers return_pct if available, falls back to pnl.

    Returns
    -------
    tuple[np.ndarray | None, str]
        Returns array and label ("return_pct", "pnl", or "none").
    """
    if trades_df.empty:
        return None, "none"

    # Prefer return_pct (normalized returns)
    if "return_pct" in trades_df.columns:
        return_pct = trades_df["return_pct"].dropna()
        if len(return_pct) > 0:
            return return_pct.to_numpy(dtype=float), "return_pct"

    # Fall back to pnl (dollar amounts)
    if "pnl" in trades_df.columns:
        pnl = trades_df["pnl"].dropna()
        if len(pnl) > 0:
            return pnl.to_numpy(dtype=float), "pnl"

    return None, "none"


def _build_patterns_df(patterns: list[dict[str, Any] | Any]) -> pd.DataFrame:
    """Build patterns DataFrame from pattern list.

    Returns DataFrame with columns:
    - cluster_id: int
    - n_trades: int
    - description: str
    - top_features: list[tuple]
    - hypothesis: str (optional)
    - actions: list[str] (optional)
    - confidence: float (optional)
    - separation_score: float (optional)
    - distinctiveness: float (optional)
    """
    records = []

    for pattern in patterns:
        if isinstance(pattern, dict):
            record = {
                "cluster_id": pattern.get("cluster_id", 0),
                "n_trades": pattern.get("n_trades", 0),
                "description": pattern.get("description", ""),
                "top_features": pattern.get("top_features", []),
                "separation_score": pattern.get("separation_score"),
                "distinctiveness": pattern.get("distinctiveness"),
                "hypothesis": pattern.get("hypothesis"),
                "actions": pattern.get("actions", []),
                "confidence": pattern.get("confidence"),
            }
        else:
            record = {
                "cluster_id": getattr(pattern, "cluster_id", 0),
                "n_trades": getattr(pattern, "n_trades", 0),
                "description": getattr(pattern, "description", ""),
                "top_features": list(getattr(pattern, "top_features", [])),
                "separation_score": getattr(pattern, "separation_score", None),
                "distinctiveness": getattr(pattern, "distinctiveness", None),
                "hypothesis": getattr(pattern, "hypothesis", None),
                "actions": list(getattr(pattern, "actions", []) or []),
                "confidence": getattr(pattern, "confidence", None),
            }
        records.append(record)

    if not records:
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "n_trades",
                "description",
                "top_features",
                "separation_score",
                "distinctiveness",
                "hypothesis",
                "actions",
                "confidence",
            ]
        )

    return pd.DataFrame(records)


def _parse_timestamp(value: Any) -> datetime | None:
    """Parse a value into datetime or None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        if not value or value == "N/A" or value == "None":
            return None
        try:
            # Try ISO format first
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            try:
                # Try common datetime formats
                for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"]:
                    try:
                        return datetime.strptime(value, fmt)
                    except ValueError:
                        continue
            except Exception:
                pass
    return None


def _safe_float(value: Any) -> float | None:
    """Safely convert value to float or None.

    Fixes the float(None) bug in the original dashboard.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
