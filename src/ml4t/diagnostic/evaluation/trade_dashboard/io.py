"""Dashboard I/O operations.

Handles loading data from uploaded files with security considerations.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ml4t.diagnostic.evaluation.trade_shap.models import TradeShapResult


class PickleDisabledError(Exception):
    """Raised when pickle loading is attempted but disabled."""

    pass


def load_result_from_upload(
    uploaded_file: Any,
    allow_pickle: bool = False,
) -> TradeShapResult | dict[str, Any]:
    """Load TradeShapResult from uploaded file.

    Parameters
    ----------
    uploaded_file : streamlit.UploadedFile
        Uploaded JSON or pickle file.
    allow_pickle : bool, default False
        Whether to allow pickle files. Disabled by default for security.
        Pickle files can execute arbitrary code when loaded.

    Returns
    -------
    TradeShapResult or dict
        Loaded result object or dictionary.

    Raises
    ------
    PickleDisabledError
        If pickle file uploaded but allow_pickle=False.
    ValueError
        If file format is unsupported or invalid.

    Security Warning
    ----------------
    Pickle files can execute arbitrary code when loaded. Only enable
    pickle loading for files from trusted sources.
    """
    filename = uploaded_file.name

    try:
        # JSON files are safe
        if filename.endswith(".json"):
            content = uploaded_file.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            data = json.loads(content)
            return data

        # Pickle files require explicit opt-in
        elif filename.endswith((".pkl", ".pickle")):
            if not allow_pickle:
                raise PickleDisabledError(
                    "Pickle files are disabled for security. "
                    "Pickle can execute arbitrary code. "
                    "Use JSON format or enable allow_pickle_upload in config."
                )

            import pickle

            data = pickle.loads(uploaded_file.read())
            return data

        else:
            raise ValueError(
                f"Unsupported file format: {filename}. Supported formats: .json, .pkl, .pickle"
            )

    except PickleDisabledError:
        raise
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}") from e
    except Exception as e:
        raise ValueError(f"Failed to load data from file: {e}") from e


def coerce_result_to_dict(
    result: TradeShapResult | dict[str, Any],
) -> dict[str, Any]:
    """Coerce TradeShapResult to dict format.

    This is a transitional function to help normalize input.
    Prefer using normalize_result() for full normalization.

    Parameters
    ----------
    result : TradeShapResult or dict
        Analysis result in either format.

    Returns
    -------
    dict
        Result as dictionary.
    """
    if isinstance(result, dict):
        return result

    # Convert TradeShapResult to dict
    return {
        "n_trades_analyzed": result.n_trades_analyzed,
        "n_trades_explained": result.n_trades_explained,
        "n_trades_failed": result.n_trades_failed,
        "explanations": [_explanation_to_dict(exp) for exp in result.explanations],
        "failed_trades": result.failed_trades,
        "error_patterns": [_pattern_to_dict(p) for p in result.error_patterns],
    }


def _explanation_to_dict(exp: Any) -> dict[str, Any]:
    """Convert explanation object to dict."""
    if isinstance(exp, dict):
        return exp

    result = {
        "trade_id": exp.trade_id,
        "timestamp": str(exp.timestamp) if hasattr(exp, "timestamp") else None,
        "shap_vector": list(exp.shap_vector) if hasattr(exp, "shap_vector") else [],
        "top_features": list(exp.top_features) if hasattr(exp, "top_features") else [],
    }

    # Include trade_metrics if available
    if hasattr(exp, "trade_metrics") and exp.trade_metrics:
        tm = exp.trade_metrics
        result["trade_metrics"] = {
            "pnl": getattr(tm, "pnl", None),
            "return_pct": getattr(tm, "return_pct", None),
            "entry_time": str(getattr(tm, "entry_time", "")) or None,
            "exit_time": str(getattr(tm, "exit_time", "")) or None,
            "duration_days": getattr(tm, "duration_days", None),
            "entry_price": getattr(tm, "entry_price", None),
            "exit_price": getattr(tm, "exit_price", None),
            "symbol": getattr(tm, "symbol", None),
        }

    return result


def _pattern_to_dict(pattern: Any) -> dict[str, Any]:
    """Convert error pattern object to dict."""
    if isinstance(pattern, dict):
        return pattern

    return {
        "cluster_id": pattern.cluster_id,
        "n_trades": pattern.n_trades,
        "description": getattr(pattern, "description", ""),
        "top_features": list(pattern.top_features) if hasattr(pattern, "top_features") else [],
        "separation_score": getattr(pattern, "separation_score", None),
        "distinctiveness": getattr(pattern, "distinctiveness", None),
        "hypothesis": getattr(pattern, "hypothesis", None),
        "actions": list(pattern.actions) if hasattr(pattern, "actions") and pattern.actions else [],
        "confidence": getattr(pattern, "confidence", None),
    }
