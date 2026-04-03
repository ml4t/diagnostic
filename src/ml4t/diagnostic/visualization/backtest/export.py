"""Export system for backtest tearsheets.

Renders selected workspaces from the same dashboard content model as the
interactive tearsheet. Supports:
- Single-workspace HTML exports (e.g., just the Trading tab)
- Multi-workspace flat exports (e.g., Overview + Performance for a report)
- Print-ready layout (no tabs, sequential sections)

All exports share the same CSS, chart styling, and content model as the
interactive dashboard — no second rendering pipeline.
"""

from __future__ import annotations

import html as html_mod
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars as pl

from .template_system import HTML_TEMPLATE, TEARSHEET_CSS, get_template

if TYPE_CHECKING:
    from ml4t.diagnostic.evaluation.factor.data import FactorData
    from ml4t.diagnostic.evaluation.trade_shap.models import TradeShapResult
    from ml4t.diagnostic.integration.backtest_profile import BacktestProfile
    from ml4t.diagnostic.integration.report_metadata import BacktestReportMetadata

WorkspaceName = Literal[
    "overview", "performance", "trading", "validation", "factors", "ml"
]

# Common workspace bundles
EXPORT_BUNDLES: dict[str, tuple[WorkspaceName, ...]] = {
    "executive": ("overview",),
    "full_report": ("overview", "performance", "trading", "validation"),
    "performance": ("overview", "performance"),
    "trading": ("overview", "trading"),
    "validation": ("overview", "validation"),
    "ml": ("overview", "ml"),
    "factors": ("overview", "factors"),
}


def export_workspaces(
    workspaces: tuple[WorkspaceName, ...] | str,
    *,
    profile: BacktestProfile | None = None,
    trades: pl.DataFrame | None = None,
    returns: pl.Series | np.ndarray | None = None,
    equity_curve: pl.DataFrame | None = None,
    metrics: dict[str, Any] | None = None,
    predictions: pl.DataFrame | None = None,
    benchmark_returns: pl.Series | np.ndarray | None = None,
    benchmark_name: str = "Benchmark",
    report_metadata: BacktestReportMetadata | None = None,
    n_trials: int | None = None,
    shap_result: TradeShapResult | None = None,
    factor_data: FactorData | None = None,
    theme: Literal["default", "dark", "print", "presentation"] = "default",
    title: str | None = None,
    subtitle: str | None = None,
    output_path: str | Path | None = None,
    include_plotlyjs: bool = True,
) -> str:
    """Export selected workspaces as a standalone HTML document.

    Parameters
    ----------
    workspaces : tuple of workspace names, or a bundle name
        Which workspaces to include. Use a tuple like
        ``("overview", "performance")`` or a bundle name like
        ``"executive"`` or ``"full_report"``.
    profile : BacktestProfile, optional
        Full backtest profile for rich analytics.
    trades, returns, equity_curve, metrics, predictions
        Raw data fallbacks (same as generate_backtest_tearsheet).
    theme : str
        Visual theme.
    output_path : str or Path, optional
        Save HTML to this path.

    Returns
    -------
    str
        Standalone HTML document with only the selected workspaces.

    Examples
    --------
    >>> html = export_workspaces(
    ...     "executive",
    ...     profile=profile,
    ...     theme="print",
    ...     output_path="overview_report.html",
    ... )
    >>> html = export_workspaces(
    ...     ("overview", "performance", "validation"),
    ...     profile=profile,
    ...     output_path="investor_report.html",
    ... )
    """
    # Resolve bundle name to workspace tuple
    if isinstance(workspaces, str):
        if workspaces not in EXPORT_BUNDLES:
            raise ValueError(
                f"Unknown bundle '{workspaces}'. "
                f"Available: {list(EXPORT_BUNDLES.keys())}"
            )
        workspace_names = EXPORT_BUNDLES[workspaces]
    else:
        workspace_names = workspaces

    # Reuse the full tearsheet pipeline but filter to selected workspaces
    from .presets import get_dashboard_preset
    from .tearsheet import (
        _compute_portfolio_summary_metrics,
        _enrich_benchmark_metrics,
        _enrich_from_trades,
        _enrich_validation_metrics,
        _generate_sections,
        _infer_periods_per_year,
        _normalize_profile_metrics,
        _plotly_js_tag,
        _resolve_report_metadata,
        _section_workspace_name,
    )

    template_name: Literal["quant_trader", "hedge_fund", "risk_manager", "full"] = "full"
    tmpl = get_template(template_name)
    preset = get_dashboard_preset(template_name)

    # Seed from profile
    if profile is not None:
        trades = trades if trades is not None else profile.trades_df
        returns = returns if returns is not None else profile.daily_returns
        equity_curve = equity_curve if equity_curve is not None else profile.equity_df
        if predictions is None and profile.has_predictions:
            predictions = profile.predictions_df
        metrics = dict(metrics or {})
        for key, value in _normalize_profile_metrics(profile).items():
            metrics.setdefault(key, value)

    periods_per_year = _infer_periods_per_year(returns)
    benchmark_metrics = _compute_portfolio_summary_metrics(
        benchmark_returns, periods_per_year=periods_per_year,
    )
    document_title, report_title, report_subtitle, resolved_benchmark_name = (
        _resolve_report_metadata(report_metadata, title, subtitle, benchmark_name)
    )

    if returns is not None and metrics is not None:
        portfolio_stats = _compute_portfolio_summary_metrics(
            returns, periods_per_year=periods_per_year,
        )
        for key, value in portfolio_stats.items():
            metrics.setdefault(key, value)
        if "sharpe_ratio" in metrics and "sharpe" not in metrics:
            metrics["sharpe"] = metrics["sharpe_ratio"]
        if "sharpe" in metrics and "sharpe_ratio" not in metrics:
            metrics["sharpe_ratio"] = metrics["sharpe"]

    if returns is not None and benchmark_returns is not None and metrics is not None:
        _enrich_benchmark_metrics(returns, benchmark_returns, metrics, periods_per_year)
    if trades is not None and metrics is not None:
        _enrich_from_trades(trades, metrics)
    if returns is not None:
        _enrich_validation_metrics(metrics or {}, returns, n_trials=n_trials)

    # Auto-enable ML/factor sections
    if predictions is not None and not predictions.is_empty():
        for section in tmpl.sections:
            if section.name in (
                "ml_summary_strip", "ic_time_series", "quintile_returns",
                "prediction_trade_alignment", "prediction_calibration",
            ):
                section.enabled = True
    if factor_data is not None:
        for section in tmpl.sections:
            if section.name.startswith("factor_"):
                section.enabled = True

    # Only enable sections that belong to the selected workspaces
    selected = set(workspace_names)
    for section in tmpl.sections:
        ws = _section_workspace_name(section.name)
        if ws not in selected:
            section.enabled = False

    sections_html = _generate_sections(
        tmpl,
        preset=preset,
        profile=profile,
        trades=trades,
        returns=returns,
        equity_curve=equity_curve,
        metrics=metrics,
        predictions=predictions,
        benchmark_returns=benchmark_returns,
        benchmark_metrics=benchmark_metrics,
        benchmark_name=resolved_benchmark_name,
        n_trials=n_trials,
        shap_result=shap_result,
        factor_data=factor_data,
        report_metadata=report_metadata,
        theme=theme,
        interactive=True,
    )

    html = HTML_TEMPLATE.format(
        theme=theme if theme == "dark" else "light",
        document_title=html_mod.escape(f"{document_title} — Export"),
        title=html_mod.escape(report_title),
        subtitle=html_mod.escape(report_subtitle),
        timestamp=datetime.now().strftime("%Y-%m-%d"),
        css=TEARSHEET_CSS,
        plotly_js=_plotly_js_tag(include_plotlyjs),
        sections_html=sections_html,
    )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html)

    return html
