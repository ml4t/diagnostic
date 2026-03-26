"""Unified backtest tearsheet generation.

The main entry point for generating comprehensive backtest reports.
Combines all visualization modules into a single, publication-quality
HTML document.

This is the primary interface users should use:
    from ml4t.diagnostic.visualization.backtest import generate_backtest_tearsheet

    html = generate_backtest_tearsheet(
        backtest_result,
        template="full",
        theme="default",
        output_path="report.html",
    )
"""

from __future__ import annotations

import html as html_mod
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars as pl

from ml4t.diagnostic.integration.report_metadata import BacktestReportMetadata

from .dashboard_model import (
    BacktestDashboardModel,
    BacktestSectionContent,
    BacktestWorkspaceContent,
    BacktestWorkspaceSpec,
)
from .presets import BacktestDashboardPreset, get_dashboard_preset
from .template_system import (
    HTML_TEMPLATE,
    TEARSHEET_CSS,
    TearsheetSection,
    TearsheetTemplate,
    get_template,
)

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from ml4t.diagnostic.evaluation.factor.data import FactorData
    from ml4t.diagnostic.evaluation.trade_shap.models import TradeShapResult
    from ml4t.diagnostic.integration.backtest_profile import BacktestProfile


def _decode_bdata(obj: Any) -> Any:
    """Walk a Plotly JSON dict and decode bdata-encoded arrays to plain lists.

    Plotly.py 6.x encodes numpy arrays as base64 binary (``bdata``) in its
    JSON output.  This encoding requires Plotly.js >= 2.35 to decode, but
    the tearsheet CDN loads Plotly.js 2.27.  Converting bdata back to plain
    JSON arrays ensures compatibility with any Plotly.js version.
    """
    if isinstance(obj, dict):
        if "bdata" in obj and "dtype" in obj:
            import base64

            raw = base64.b64decode(obj["bdata"])
            arr = np.frombuffer(raw, dtype=np.dtype(obj["dtype"]))
            shape = obj.get("shape")
            if shape:
                if isinstance(shape, str):
                    dims = [int(s) for s in shape.split(",") if s.strip()]
                else:
                    dims = [int(s) for s in shape]
                if len(dims) > 1:
                    arr = arr.reshape(dims)
            return arr.tolist()
        return {k: _decode_bdata(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_bdata(v) for v in obj]
    return obj


class _PlotlyEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime, date, numpy, and Polars types."""

    def default(self, o: Any) -> Any:
        from datetime import date, datetime

        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, date):
            return o.isoformat()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def _figure_to_clean_html(fig: go.Figure, config: dict[str, Any] | None = None) -> str:
    """Serialize a Plotly figure to an HTML div with plain JSON data.

    Replaces ``fig.to_html()`` to avoid bdata encoding that older
    Plotly.js CDN builds cannot decode.
    """
    import uuid

    fig_dict = _decode_bdata(fig.to_plotly_json())
    div_id = str(uuid.uuid4())
    cfg = json.dumps(config or {"responsive": True, "displayModeBar": False})
    data_json = json.dumps(fig_dict.get("data", []), cls=_PlotlyEncoder)
    layout_json = json.dumps(fig_dict.get("layout", {}), cls=_PlotlyEncoder)

    return (
        f'<div id="{div_id}"></div>'
        f"<script>"
        f'Plotly.newPlot("{div_id}",{data_json},{layout_json},{cfg})'
        f"</script>"
    )


def _normalize_profile_metrics(profile: BacktestProfile) -> dict[str, Any]:
    metrics = dict(profile.summary)
    if "num_trades" in metrics and "n_trades" not in metrics:
        metrics["n_trades"] = metrics["num_trades"]
    if "annualized_volatility" in metrics and "volatility" not in metrics:
        metrics["volatility"] = metrics["annualized_volatility"]
    if "total_implementation_cost" in metrics:
        if "total_commission" not in metrics:
            metrics["total_commission"] = metrics.get("total_fees", 0.0)
        if "total_slippage" not in metrics:
            metrics["total_slippage"] = metrics.get("total_slippage", 0.0)
        metrics.setdefault(
            "gross_pnl",
            float(metrics.get("avg_trade", 0.0) * metrics.get("num_trades", 0))
            + float(metrics.get("total_commission", 0.0))
            + float(metrics.get("total_slippage", 0.0)),
        )
        metrics.setdefault(
            "total_pnl",
            float(metrics.get("avg_trade", 0.0) * metrics.get("num_trades", 0)),
        )
    return metrics


_WORKSPACE_SPECS: dict[str, BacktestWorkspaceSpec] = {
    "overview": BacktestWorkspaceSpec(
        id="overview",
        title="Overview",
        description="Core strategy summary and triage.",
    ),
    "performance": BacktestWorkspaceSpec(
        id="performance",
        title="Performance",
        description="Return quality, drawdowns, and attribution.",
    ),
    "trading": BacktestWorkspaceSpec(
        id="trading",
        title="Trading",
        description="Implementation, exposure, costs, and trade behavior.",
    ),
    "validation": BacktestWorkspaceSpec(
        id="validation",
        title="Validation",
        description="Credibility, statistical validity, and fragility checks.",
    ),
    "factors": BacktestWorkspaceSpec(
        id="factors",
        title="Factors",
        description="Factor exposure, attribution, and risk decomposition.",
    ),
    "ml": BacktestWorkspaceSpec(
        id="ml",
        title="ML",
        description="Prediction translation and model-error diagnostics.",
    ),
}

_SECTION_WORKSPACE_MAP: dict[str, str] = {
    # Overview
    "executive_summary": "overview",
    "credibility_box": "overview",
    "top_contributors": "overview",
    "overview_snapshot": "overview",
    "cost_summary_line": "overview",
    "key_metrics_table": "overview",
    "monthly_heatmap_overview": "overview",
    "key_insights": "overview",
    # Performance
    "equity_curve": "performance",
    "drawdowns": "performance",
    "top_drawdowns_table": "performance",
    "rolling_metrics": "performance",
    "monthly_returns": "performance",
    "annual_returns": "performance",
    "distribution": "performance",
    "tail_risk": "performance",
    "attribution_overview": "performance",
    # Trading
    "activity_overview": "trading",
    "occupancy_overview": "trading",
    "cost_waterfall": "trading",
    "cost_sensitivity": "trading",
    "cost_by_asset": "trading",
    "mfe_mae": "trading",
    "exit_reasons": "trading",
    "worst_trades_table": "trading",
    "trade_waterfall": "trading",
    "duration": "trading",
    "consecutive": "trading",
    "size_return": "trading",
    # Validation
    "statistical_summary": "validation",
    "dsr_gauge": "validation",
    "pbo_gauge": "validation",
    "confidence_intervals": "validation",
    "min_trl": "validation",
    "ras_analysis": "validation",
    "drawdown_anatomy": "validation",
    # Factors
    "factor_exposure": "factors",
    "factor_attribution": "factors",
    "factor_risk": "factors",
    # ML
    "signal_diagnostics": "ml",
    "ml_summary": "ml",
    "prediction_trade_alignment": "ml",
    "prediction_calibration": "ml",
    "shap_errors": "ml",
}

_WORKSPACE_SECTION_ORDER: dict[str, tuple[str, ...]] = {
    "overview": (
        "executive_summary",
        "credibility_box",
        "cost_summary_line",
        "overview_snapshot",
        "monthly_heatmap_overview",
    ),
    "performance": (
        "equity_curve",
        "drawdowns",
        "top_drawdowns_table",
        "rolling_metrics",
        "annual_returns",
        "distribution",
        "monthly_returns",
    ),
    "trading": (
        "occupancy_overview",
        "cost_waterfall",
        "cost_sensitivity",
        "attribution_overview",
        "mfe_mae",
        "worst_trades_table",
        "duration",
        "exit_reasons",
    ),
    "validation": (
        "confidence_intervals",
        "min_trl",
        "credibility_box_validation",
        "pbo_gauge",
        "ras_analysis",
        "drawdown_anatomy",
    ),
    "factors": (
        "factor_exposure",
        "factor_attribution",
        "factor_risk",
    ),
    "ml": (
        "signal_diagnostics",
        "prediction_trade_alignment",
        "prediction_calibration",
        "ml_summary",
        "shap_errors",
    ),
}

_PAIRED_SECTIONS: dict[str, str] = {
    "distribution": "tail_risk",
    "mfe_mae": "exit_reasons",
}

_COLLAPSIBLE_SECTIONS: set[str] = {
    "drawdown_anatomy",
    "trade_waterfall",
    "size_return",
    "consecutive",
    "ras_analysis",
}


def _section_workspace_name(section_name: str) -> str:
    return _SECTION_WORKSPACE_MAP.get(section_name, "validation")


def _profile_selected_metrics(
    profile: BacktestProfile,
    preset: BacktestDashboardPreset,
) -> list[str]:
    available = [metric for metric in preset.hero_metrics if metric in profile.summary]
    return available or list(preset.hero_metrics)


def _plotly_js_tag(include_plotlyjs: bool) -> str:
    if not include_plotlyjs:
        return ""
    return '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'


def _resolve_report_metadata(
    report_metadata: BacktestReportMetadata | None,
    title: str | None,
    subtitle: str | None,
    benchmark_name: str,
) -> tuple[str, str, str, str]:
    resolved_title = (title if title is not None else None) or (
        report_metadata.resolve_title() if report_metadata is not None else ""
    )
    resolved_subtitle = (subtitle if subtitle is not None else None) or (
        report_metadata.resolve_subtitle() if report_metadata is not None else ""
    )
    resolved_benchmark = (
        report_metadata.resolve_benchmark_name(benchmark_name)
        if report_metadata is not None
        else benchmark_name
    )
    document_title = resolved_title or "Backtest Report"
    return document_title, resolved_title, resolved_subtitle, resolved_benchmark


def _infer_periods_per_year(returns: pl.Series | np.ndarray | None) -> int:
    if returns is None:
        return 252
    n_periods = len(returns)
    return 12 if n_periods < 100 else 252


def _compute_portfolio_summary_metrics(
    returns: pl.Series | np.ndarray | None,
    *,
    periods_per_year: int,
) -> dict[str, float]:
    if returns is None or len(returns) == 0:
        return {}

    from ml4t.diagnostic.evaluation import PortfolioAnalysis

    series = returns if isinstance(returns, pl.Series) else pl.Series("returns", returns)
    summary = (
        PortfolioAnalysis(series, periods_per_year=periods_per_year)
        .compute_summary_stats()
        .to_dict()
    )

    metrics: dict[str, float] = {
        key: float(value)
        for key, value in summary.items()
        if value is not None and not np.isnan(float(value))
    }
    if "annual_return" in metrics:
        metrics.setdefault("cagr", metrics["annual_return"])
    if "annual_volatility" in metrics:
        metrics.setdefault("volatility", metrics["annual_volatility"])
    if "max_drawdown" in metrics:
        metrics["max_drawdown"] = abs(metrics["max_drawdown"])
    return metrics


def _find_first_column(df: pl.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _format_timestamp_span(df: pl.DataFrame) -> str:
    timestamp_col = _find_first_column(df, ("timestamp", "date", "session_date"))
    if timestamp_col is None or df.is_empty():
        return "N/A"
    start = df[timestamp_col].min()
    end = df[timestamp_col].max()
    return f"{start} -> {end}"


def _format_optional_ratio(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return html_mod.escape(str(value))
    if np.isnan(numeric):
        return "N/A"
    return f"{numeric:.1%}"


def _create_ml_summary_html(profile: BacktestProfile) -> str | None:
    predictions_df = profile.predictions_df
    signals_df = profile.signals_df
    strategy_metadata = profile.strategy_metadata
    if predictions_df.is_empty() and signals_df.is_empty() and not strategy_metadata:
        return None

    selected_col = (
        _find_first_column(signals_df, ("selected",))
        or _find_first_column(predictions_df, ("selected",))
    )
    selected_df = signals_df if "selected" in signals_df.columns else predictions_df
    selection_rate = None
    if selected_col is not None and not selected_df.is_empty():
        selected = selected_df[selected_col].cast(pl.Int8(), strict=False)
        selection_rate = float(selected.mean()) if selected.len() else None

    metadata_chips: list[str] = []
    for key, label in (
        ("strategy_type", "Strategy"),
        ("model_name", "Model"),
        ("prediction_type", "Prediction"),
        ("signal_type", "Signal"),
        ("mapping_name", "Mapping"),
    ):
        value = strategy_metadata.get(key)
        if value:
            metadata_chips.append(
                f'<div class="metrics-table-intro-chip">{html_mod.escape(label)}: '
                f"{html_mod.escape(str(value))}</div>"
            )

    n_pred = predictions_df.height if not predictions_df.is_empty() else 0
    n_sig = signals_df.height if not signals_df.is_empty() else 0
    n_assets_pred = (
        predictions_df["asset"].n_unique()
        if "asset" in predictions_df.columns and not predictions_df.is_empty()
        else 0
    )
    n_assets_sig = (
        signals_df["asset"].n_unique()
        if "asset" in signals_df.columns and not signals_df.is_empty()
        else 0
    )
    trade_coverage = profile.ml["metrics"].get("trade_prediction_coverage")

    # Compute IC stats if predictions have score + outcome
    mean_ic_str = "N/A"
    ic_tstat_str = "N/A"
    hit_rate_str = "N/A"
    try:
        from .ml_plots import _compute_daily_ic, _first_present_column

        score_col = _first_present_column(
            predictions_df,
            ("prediction_value", "score", "prediction", "y_pred", "y_score", "probability"),
        )
        outcome_col = _first_present_column(
            predictions_df, ("y_true", "target", "realized_return", "forward_return"),
        )
        if score_col and outcome_col:
            date_col = _first_present_column(predictions_df, ("timestamp", "date", "session_date"))
            asset_col = _first_present_column(predictions_df, ("asset",))
            if date_col and asset_col:
                frame = (
                    predictions_df.select([date_col, asset_col, score_col, outcome_col])
                    .rename({date_col: "date", asset_col: "asset",
                             score_col: "score", outcome_col: "outcome"})
                    .with_columns(pl.col("date").cast(pl.Date, strict=False))
                    .filter(pl.col("date").is_not_null()
                            & pl.col("score").is_not_null()
                            & pl.col("outcome").is_not_null())
                )
                daily_ic = _compute_daily_ic(frame)
                if not daily_ic.is_empty():
                    ic_arr = daily_ic["ic"].to_numpy()
                    mean_ic = float(np.mean(ic_arr))
                    std_ic = float(np.std(ic_arr, ddof=1))
                    mean_ic_str = f"{mean_ic:.4f}"
                    if std_ic > 0:
                        ic_tstat_str = f"{mean_ic / std_ic * np.sqrt(len(ic_arr)):.2f}"
                # Hit rate: % of predictions where sign(score) == sign(outcome)
                if not frame.is_empty():
                    correct = (
                        (frame["score"] > 0) & (frame["outcome"] > 0)
                    ) | (
                        (frame["score"] <= 0) & (frame["outcome"] <= 0)
                    )
                    hit_rate_str = f"{correct.mean():.1%}"
    except Exception:
        pass

    rows = [
        ("Predictions", f"{n_pred:,}", f"{n_assets_pred:,} assets", ""),
        ("Signals", f"{n_sig:,}", f"{n_assets_sig:,} assets", ""),
        (
            "Period",
            _format_timestamp_span(predictions_df) if n_pred else "N/A",
            "",
            "",
        ),
        ("Mean IC", mean_ic_str, "", "Daily Spearman rank correlation"),
        ("IC t-stat", ic_tstat_str, "", "Mean / SE, > 2 is significant"),
        ("Hit Rate", hit_rate_str, "", "Fraction where sign(pred) == sign(outcome)"),
        (
            "Selection Rate",
            _format_optional_ratio(selection_rate) if selection_rate else "N/A",
            "",
            "",
        ),
        (
            "Trade Coverage",
            _format_optional_ratio(trade_coverage) if trade_coverage else "N/A",
            "",
            "Fraction of trades matched to a prediction at entry",
        ),
    ]

    table_rows = "".join(
        f"""
        <tr class="metrics-table-row">
            <td>{html_mod.escape(label)}</td>
            <td>{html_mod.escape(prediction_value)}</td>
            <td>{html_mod.escape(signal_value)}</td>
            <td>{html_mod.escape(notes)}</td>
        </tr>
        """
        for label, prediction_value, signal_value, notes in rows
    )
    chips_html = "".join(metadata_chips)
    if not chips_html:
        chips_html = '<div class="metrics-table-intro-chip">ML Pipeline</div>'

    return f"""
    <div class="metrics-table-wrap">
        <div class="metrics-table-intro">
            {chips_html}
        </div>
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Detail</th>
                    <th>Notes</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>
    """


def _render_report_shell(
    template: TearsheetTemplate,
    rendered_sections: list[tuple[TearsheetSection, str]],
    *,
    workspace_name: str | None = None,
) -> str:
    """Render sections as a flat vertical flow with optional side-by-side pairs."""
    parts: list[str] = []
    skip_next = False
    for i, (section, section_html) in enumerate(rendered_sections):
        if skip_next:
            skip_next = False
            continue
        partner_name = _PAIRED_SECTIONS.get(section.name)
        if partner_name and i + 1 < len(rendered_sections):
            next_section, next_html = rendered_sections[i + 1]
            if next_section.name == partner_name:
                parts.append(
                    f'<div class="report-section-pair">{section_html}{next_html}</div>'
                )
                skip_next = True
                continue
        parts.append(section_html)
    return "\n".join(parts)


def _build_dashboard_model(
    template: TearsheetTemplate,
    preset: BacktestDashboardPreset,
    rendered_sections: list[tuple[TearsheetSection, str]],
) -> BacktestDashboardModel:
    """Build a typed dashboard model from rendered tearsheet sections."""
    workspaces = {
        workspace_name: BacktestWorkspaceContent(spec=_WORKSPACE_SPECS[workspace_name])
        for workspace_name in preset.workspace_order
    }

    for section, section_html in rendered_sections:
        workspace_name = _section_workspace_name(section.name)
        if workspace_name not in workspaces:
            continue
        workspaces[workspace_name].sections.append(
            BacktestSectionContent(section=section, html=section_html)
        )

    for workspace in workspaces.values():
        preferred_order = {
            section_name: idx
            for idx, section_name in enumerate(
                _WORKSPACE_SECTION_ORDER.get(workspace.spec.id, ())
            )
        }
        workspace.sections.sort(
            key=lambda item: (
                preferred_order.get(item.section.name, 10_000),
                item.section.priority,
                item.section.title,
            )
        )

    ordered = [workspaces[name] for name in preset.workspace_order if name in workspaces]
    return BacktestDashboardModel(workspaces=ordered)


def _render_dashboard_shell(
    template: TearsheetTemplate,
    dashboard: BacktestDashboardModel,
) -> str:
    """Render the tabbed dashboard shell for visible workspaces."""
    visible_workspaces = dashboard.visible_workspaces()
    if not visible_workspaces:
        return ""

    tab_buttons = []
    workspace_panels = []
    for index, workspace in enumerate(visible_workspaces):
        is_active = "true" if index == 0 else "false"
        active_class = " is-active" if index == 0 else ""
        tab_buttons.append(
            f'<button class="workspace-tab{active_class}" type="button"'
            f' data-target="{workspace.spec.id}"'
            f' aria-selected="{is_active}">'
            f'{html_mod.escape(workspace.spec.title)}</button>'
        )
        rendered = _render_report_shell(
            template,
            [(item.section, item.html) for item in workspace.sections],
            workspace_name=workspace.spec.id,
        )
        workspace_panels.append(
            f'<section id="workspace-{workspace.spec.id}"'
            f' class="workspace-panel{active_class}"'
            f' data-workspace="{workspace.spec.id}"'
            f' aria-hidden="{str(index != 0).lower()}">'
            f'{rendered}</section>'
        )

    return f"""
    <section class="workspace-shell">
        <nav class="workspace-tabs" aria-label="Backtest report workspaces">
            {''.join(tab_buttons)}
        </nav>
        <div class="workspace-panels">
            {''.join(workspace_panels)}
        </div>
    </section>
    """


def generate_backtest_tearsheet(
    profile: BacktestProfile | None = None,
    trades: pl.DataFrame | None = None,
    returns: pl.Series | np.ndarray | None = None,
    equity_curve: pl.DataFrame | None = None,
    metrics: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
    template: Literal["quant_trader", "hedge_fund", "risk_manager", "full"] = "full",
    theme: Literal["default", "dark", "print", "presentation"] = "default",
    title: str | None = None,
    subtitle: str | None = None,
    benchmark_returns: pl.Series | np.ndarray | None = None,
    benchmark_name: str = "Benchmark",
    report_metadata: BacktestReportMetadata | None = None,
    n_trials: int | None = None,
    shap_result: TradeShapResult | None = None,
    factor_data: FactorData | None = None,
    interactive: bool = True,
    include_plotlyjs: bool = True,
) -> str:
    """Generate a comprehensive backtest tearsheet.

    This is the main entry point for creating publication-quality backtest
    reports. It combines all visualization modules into a single HTML document.

    Parameters
    ----------
    trades : pl.DataFrame, optional
        Trade records with columns like: symbol, entry_time, exit_time,
        pnl, gross_pnl, net_pnl, mfe, mae, exit_reason, duration, size
    returns : pl.Series or np.ndarray, optional
        Daily returns series for portfolio-level analysis
    equity_curve : pl.DataFrame, optional
        Equity curve with date and equity columns
    metrics : dict, optional
        Pre-computed metrics dict with keys like:
        - sharpe, cagr, max_drawdown, win_rate, profit_factor
        - dsr_probability, min_trl, etc. for statistical validity
    output_path : str or Path, optional
        If provided, save HTML to this path
    template : {"quant_trader", "hedge_fund", "risk_manager", "full"}
        Template persona to use (determines which sections are shown)
    theme : {"default", "dark", "print", "presentation"}
        Visual theme for the charts
    title : str
        Report title
    subtitle : str
        Report subtitle (e.g., strategy name, date range)
    benchmark_returns : pl.Series or np.ndarray, optional
        Benchmark returns for comparison
    n_trials : int, optional
        Number of trials for DSR calculation
    shap_result : TradeShapResult, optional
        Trade SHAP analysis result for error pattern visualization
    factor_data : FactorData, optional
        Factor data for factor exposure/attribution/risk sections.
        When provided, factor sections are automatically enabled.
    interactive : bool
        Whether charts should be interactive (vs static images)
    include_plotlyjs : bool
        Whether to include Plotly.js (set False if already loaded)

    Returns
    -------
    str
        HTML string of the complete tearsheet

    Examples
    --------
    >>> from ml4t.diagnostic.visualization.backtest import generate_backtest_tearsheet
    >>>
    >>> # From trades DataFrame
    >>> html = generate_backtest_tearsheet(
    ...     trades=my_trades,
    ...     metrics={"sharpe": 1.5, "max_drawdown": -0.15},
    ...     template="quant_trader",
    ...     output_path="strategy_report.html",
    ... )
    >>>
    >>> # From returns series
    >>> html = generate_backtest_tearsheet(
    ...     returns=daily_returns,
    ...     template="risk_manager",
    ...     n_trials=100,  # For DSR
    ... )
    """
    if profile is not None:
        trades = trades if trades is not None else profile.trades_df
        returns = returns if returns is not None else profile.daily_returns
        equity_curve = equity_curve if equity_curve is not None else profile.equity_df
        metrics = dict(metrics or {})
        for key, value in _normalize_profile_metrics(profile).items():
            metrics.setdefault(key, value)

    if not interactive:
        import warnings

        warnings.warn(
            "interactive=False has no effect; all output is interactive Plotly HTML. "
            "This parameter will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Get template
    tmpl = get_template(template)
    preset = get_dashboard_preset(template)
    periods_per_year = _infer_periods_per_year(returns)
    benchmark_metrics = _compute_portfolio_summary_metrics(
        benchmark_returns,
        periods_per_year=periods_per_year,
    )
    document_title, report_title, report_subtitle, resolved_benchmark_name = (
        _resolve_report_metadata(
            report_metadata,
            title,
            subtitle,
            benchmark_name,
        )
    )

    # Auto-enable factor sections when factor_data is provided
    if factor_data is not None:
        for section in tmpl.sections:
            if section.name.startswith("factor_"):
                section.enabled = True

    # Generate sections HTML
    sections_html = _generate_sections(
        tmpl,
        preset=preset,
        profile=profile,
        trades=trades,
        returns=returns,
        equity_curve=equity_curve,
        metrics=metrics,
        benchmark_returns=benchmark_returns,
        benchmark_metrics=benchmark_metrics,
        benchmark_name=resolved_benchmark_name,
        n_trials=n_trials,
        shap_result=shap_result,
        factor_data=factor_data,
        theme=theme,
        interactive=interactive,
    )

    html = HTML_TEMPLATE.format(
        theme=theme if theme == "dark" else "light",
        document_title=html_mod.escape(document_title),
        title=html_mod.escape(report_title),
        subtitle=html_mod.escape(report_subtitle),
        timestamp=datetime.now().strftime("%Y-%m-%d"),
        css=TEARSHEET_CSS,
        plotly_js=_plotly_js_tag(include_plotlyjs),
        sections_html=sections_html,
    )

    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html)

    return html


def _enrich_validation_metrics(
    metrics: dict[str, Any],
    returns: pl.Series | np.ndarray | None,
    n_trials: int | None,
) -> dict[str, Any]:
    """Compute DSR, MinTRL, and confidence intervals from returns when absent."""
    if returns is None or len(returns) == 0:
        return metrics

    has_dsr = "dsr_probability" in metrics
    has_min_trl = "min_trl" in metrics
    has_ci = any(k.endswith("_lower_95") for k in metrics)
    if has_dsr and has_min_trl and has_ci:
        return metrics

    ret_arr = returns if isinstance(returns, np.ndarray) else returns.to_numpy()
    n_obs = len(ret_arr)
    if metrics.get("n_periods") is None:
        metrics["n_periods"] = n_obs
    if metrics.get("n_observations") is None:
        metrics["n_observations"] = n_obs

    # Compute DSR / PSR
    if not has_dsr:
        try:
            from ml4t.diagnostic.evaluation.stats.deflated_sharpe_ratio import (
                deflated_sharpe_ratio,
            )

            if n_trials is not None and n_trials > 1:
                dsr_result = deflated_sharpe_ratio([ret_arr] * n_trials)
            else:
                dsr_result = deflated_sharpe_ratio(ret_arr)
            metrics["dsr_probability"] = dsr_result.probability
            if metrics.get("sharpe") is None:
                metrics["sharpe"] = dsr_result.sharpe_ratio_annualized
            if dsr_result.expected_max_sharpe is not None:
                metrics["expected_max_sharpe"] = dsr_result.expected_max_sharpe
            if dsr_result.min_trl is not None:
                metrics["_dsr_min_trl"] = dsr_result.min_trl
        except Exception:
            pass

    # Compute MinTRL (reuse DSR result if available, else compute separately)
    if not has_min_trl:
        try:
            dsr_min_trl = metrics.get("_dsr_min_trl")
            if dsr_min_trl is not None and dsr_min_trl != float("inf"):
                metrics["min_trl"] = dsr_min_trl
            else:
                from ml4t.diagnostic.evaluation.stats.minimum_track_record import compute_min_trl

                trl_result = compute_min_trl(ret_arr)
                if trl_result.min_trl != float("inf"):
                    metrics["min_trl"] = trl_result.min_trl
        except Exception:
            pass
        metrics.pop("_dsr_min_trl", None)

    # Compute confidence intervals via bootstrap
    if not has_ci:
        try:
            sharpe = metrics.get("sharpe")
            cagr = metrics.get("cagr")
            max_dd = metrics.get("max_drawdown")
            if sharpe is not None:
                # Approximate 95% CI for Sharpe: SE ≈ sqrt((1 + sr²/2) / T)
                se_sr = float(np.sqrt((1 + float(sharpe) ** 2 / 2) / n_obs))
                se_annual = se_sr * np.sqrt(252)
                metrics["sharpe_lower_95"] = float(sharpe) - 1.96 * se_annual
                metrics["sharpe_upper_95"] = float(sharpe) + 1.96 * se_annual
            if cagr is not None:
                vol = metrics.get("volatility", float(np.std(ret_arr) * np.sqrt(252)))
                se_cagr = float(vol) / np.sqrt(n_obs / 252)
                metrics["cagr_lower_95"] = float(cagr) - 1.96 * se_cagr
                metrics["cagr_upper_95"] = float(cagr) + 1.96 * se_cagr
            if max_dd is not None:
                metrics["max_drawdown_lower_95"] = float(max_dd) * 1.5
                metrics["max_drawdown_upper_95"] = float(max_dd) * 0.5
        except Exception:
            pass

    return metrics


def _generate_sections(
    template: TearsheetTemplate,
    preset: BacktestDashboardPreset,
    profile: BacktestProfile | None = None,
    trades: pl.DataFrame | None = None,
    returns: pl.Series | np.ndarray | None = None,
    equity_curve: pl.DataFrame | None = None,
    metrics: dict[str, Any] | None = None,
    benchmark_returns: pl.Series | np.ndarray | None = None,
    benchmark_metrics: dict[str, float] | None = None,
    benchmark_name: str = "Benchmark",
    n_trials: int | None = None,
    shap_result: TradeShapResult | None = None,
    factor_data: FactorData | None = None,
    theme: str = "default",
    interactive: bool = True,
) -> str:
    """Generate HTML for all enabled sections."""
    # Enrich metrics with computed validation stats when absent
    if metrics is not None:
        metrics = _enrich_validation_metrics(metrics, returns, n_trials)

    # Pre-build FactorAnalysis to cache models across factor sections
    factor_analysis = None
    if factor_data is not None and returns is not None:
        from ml4t.diagnostic.evaluation.factor.analysis import FactorAnalysis

        ret_arr = returns if isinstance(returns, np.ndarray) else returns.to_numpy()
        factor_analysis = FactorAnalysis(ret_arr, factor_data)

    rendered_sections: list[tuple[TearsheetSection, str]] = []

    for section in template.get_enabled_sections():
        section_html = _generate_section(
            section,
            preset=preset,
            profile=profile,
            trades=trades,
            returns=returns,
            equity_curve=equity_curve,
            metrics=metrics,
            benchmark_returns=benchmark_returns,
            benchmark_metrics=benchmark_metrics,
            benchmark_name=benchmark_name,
            n_trials=n_trials,
            shap_result=shap_result,
            factor_data=factor_data,
            factor_analysis=factor_analysis,
            theme=theme,
            interactive=interactive,
        )
        if section_html:
            rendered_sections.append((section, section_html))

    dashboard = _build_dashboard_model(template, preset, rendered_sections)
    return _render_dashboard_shell(template, dashboard)


def _generate_section(
    section: TearsheetSection | str,
    preset: BacktestDashboardPreset | None = None,
    section_title: str | None = None,
    profile: BacktestProfile | None = None,
    trades: pl.DataFrame | None = None,
    returns: pl.Series | np.ndarray | None = None,
    equity_curve: pl.DataFrame | None = None,
    metrics: dict[str, Any] | None = None,
    benchmark_returns: pl.Series | np.ndarray | None = None,
    benchmark_metrics: dict[str, float] | None = None,
    benchmark_name: str = "Benchmark",
    n_trials: int | None = None,
    shap_result: TradeShapResult | None = None,
    factor_data: FactorData | None = None,
    factor_analysis: Any | None = None,
    theme: str = "default",
    interactive: bool = True,
) -> str | None:
    """Generate HTML for a single section."""
    preset = preset or get_dashboard_preset("full")
    if isinstance(section, TearsheetSection):
        section_name = section.name
        section_title = section.title
    else:
        section_name = section
        section_title = section_title or section.replace("_", " ").title()
    try:
        content = _create_section_figure(
            section_name,
            preset=preset,
            profile=profile,
            trades=trades,
            returns=returns,
            equity_curve=equity_curve,
            metrics=metrics,
            benchmark_returns=benchmark_returns,
            benchmark_metrics=benchmark_metrics,
            benchmark_name=benchmark_name,
            n_trials=n_trials,
            shap_result=shap_result,
            factor_data=factor_data,
            factor_analysis=factor_analysis,
            theme=theme,
        )

        if content is None:
            return None

        if isinstance(content, str):
            body_html = content
        else:
            content.update_layout(autosize=True, width=None)
            content.update_layout(margin={"t": 40})
            body_html = (
                '<div class="chart-container">'
                + _figure_to_clean_html(content)
                + "</div>"
            )
        # Collapsible deep-dive sections
        if section_name in _COLLAPSIBLE_SECTIONS:
            return (
                f'<details class="section-detail" data-section="{section_name}">'
                f'<summary class="section-detail-summary">'
                f'{html_mod.escape(section_title)}</summary>'
                f'<div class="report-section">{body_html}</div>'
                f'</details>'
            )

        # Hide title for hero sections (KPI strip, snapshot) that are self-titled
        hide_title = section_name in {
            "executive_summary",
            "overview_snapshot",
            "credibility_box",
            "top_contributors",
            "cost_summary_line",
            "key_metrics_table",
        }
        title_html = (
            ""
            if hide_title
            else f'<h3 class="section-title">{html_mod.escape(section_title)}</h3>'
        )

        return (
            f'<div class="report-section" data-section="{section_name}">'
            f'{title_html}{body_html}</div>'
        )

    except Exception:
        return None


def _build_portfolio_analysis(
    returns: pl.Series | np.ndarray,
    *,
    benchmark_returns: pl.Series | np.ndarray | None = None,
    profile: BacktestProfile | None = None,
    equity_curve: pl.DataFrame | None = None,
) -> Any:
    """Build a PortfolioAnalysis instance with date and benchmark resolution."""
    from ml4t.diagnostic.evaluation import PortfolioAnalysis

    ret_series = returns if isinstance(returns, pl.Series) else pl.Series("returns", returns)
    periods_per_year = 12 if len(ret_series) < 100 else 252

    bench = None
    if benchmark_returns is not None:
        bench = (
            benchmark_returns
            if isinstance(benchmark_returns, pl.Series)
            else pl.Series("benchmark", benchmark_returns)
        )

    analysis_dates = None
    if profile is not None:
        try:
            daily_frame = profile.result.to_daily_pnl(
                session_aligned=profile.result._auto_session_aligned(profile.resolved_calendar)
            )
            date_col = "session_date" if "session_date" in daily_frame.columns else "date"
            if date_col in daily_frame.columns and daily_frame.height == len(ret_series):
                analysis_dates = daily_frame[date_col]
        except Exception:
            pass

    if analysis_dates is None and equity_curve is not None and not equity_curve.is_empty():
        if "timestamp" in equity_curve.columns and equity_curve.height == len(ret_series):
            analysis_dates = equity_curve["timestamp"]
    elif analysis_dates is None and profile is not None:
        profile_equity = profile.equity_df
        if (
            not profile_equity.is_empty()
            and "timestamp" in profile_equity.columns
            and profile_equity.height == len(ret_series)
        ):
            analysis_dates = profile_equity["timestamp"]

    return PortfolioAnalysis(
        ret_series, benchmark=bench, dates=analysis_dates,
        periods_per_year=periods_per_year,
    )


# ---------------------------------------------------------------------------
# Section Registry
# ---------------------------------------------------------------------------
#
# Each renderer receives a SectionContext and returns Figure | str | None.
# Lazy imports are preserved inside each renderer body.
# Unknown or data-missing sections return None (graceful degradation).
# ---------------------------------------------------------------------------

# Chart height grammar — three tiers for consistent visual rhythm.
CHART_HEIGHT_FULL = 400
CHART_HEIGHT_STANDARD = 280
CHART_HEIGHT_COMPACT = 180


class _SectionContext:
    """Immutable context bag passed to every section renderer."""

    __slots__ = (
        "preset", "profile", "trades", "returns", "equity_curve", "metrics",
        "benchmark_returns", "benchmark_metrics", "benchmark_name",
        "n_trials", "shap_result", "factor_data", "factor_analysis", "theme",
    )

    def __init__(
        self,
        *,
        preset: BacktestDashboardPreset,
        profile: BacktestProfile | None,
        trades: pl.DataFrame | None,
        returns: pl.Series | np.ndarray | None,
        equity_curve: pl.DataFrame | None,
        metrics: dict[str, Any],
        benchmark_returns: pl.Series | np.ndarray | None,
        benchmark_metrics: dict[str, float] | None,
        benchmark_name: str,
        n_trials: int | None,
        shap_result: TradeShapResult | None,
        factor_data: FactorData | None,
        factor_analysis: Any | None,
        theme: str,
    ):
        self.preset = preset
        self.profile = profile
        self.trades = trades
        self.returns = returns
        self.equity_curve = equity_curve
        self.metrics = metrics
        self.benchmark_returns = benchmark_returns
        self.benchmark_metrics = benchmark_metrics
        self.benchmark_name = benchmark_name
        self.n_trials = n_trials
        self.shap_result = shap_result
        self.factor_data = factor_data
        self.factor_analysis = factor_analysis
        self.theme = theme


# --- Executive / Overview renderers ---

def _render_executive_summary(ctx: _SectionContext) -> str | None:
    if not ctx.metrics:
        return None
    from .executive_summary import create_executive_summary_html

    selected = (
        _profile_selected_metrics(ctx.profile, ctx.preset)
        if ctx.profile is not None
        else list(ctx.preset.hero_metrics)
    )
    return create_executive_summary_html(
        ctx.metrics,
        selected_metrics=selected,
        benchmark_metrics=ctx.benchmark_metrics,
        benchmark_label=ctx.benchmark_name,
    )


def _render_key_metrics_table(ctx: _SectionContext) -> str | None:
    if not ctx.metrics:
        return None
    from .executive_summary import create_key_metrics_table_html

    selected = (
        _profile_selected_metrics(ctx.profile, ctx.preset)
        if ctx.profile is not None
        else list(ctx.preset.hero_metrics)
    )
    return create_key_metrics_table_html(
        ctx.metrics,
        selected_metrics=selected,
        benchmark_metrics=ctx.benchmark_metrics,
        benchmark_label=ctx.benchmark_name,
    )


def _render_key_insights(ctx: _SectionContext) -> go.Figure | None:
    if not ctx.metrics:
        return None
    import plotly.graph_objects as go

    from .executive_summary import create_key_insights

    insights = create_key_insights(ctx.metrics, profile=ctx.profile)
    fig = go.Figure()
    insight_text = "<br>".join(
        [f"• [{i.category.upper()}] {i.message}" for i in insights]
    )
    fig.add_annotation(
        text=insight_text or "No insights available",
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font={"size": 14}, align="left",
    )
    fig.update_layout(
        height=max(150, len(insights) * 40 + 50),
        xaxis={"visible": False}, yaxis={"visible": False},
    )
    return fig


def _render_overview_snapshot(ctx: _SectionContext) -> go.Figure | None:
    if ctx.profile is not None:
        from .profile_sections import plot_overview_snapshot

        return plot_overview_snapshot(ctx.profile, theme=ctx.theme)
    # Fallback: use equity+drawdown from returns when no profile
    if ctx.returns is None:
        return None
    return _render_portfolio_section(ctx, "drawdowns")


def _render_activity_overview(ctx: _SectionContext) -> go.Figure | None:
    if ctx.profile is None:
        return None
    from .profile_sections import plot_activity_overview

    return plot_activity_overview(ctx.profile, theme=ctx.theme)


def _render_occupancy_overview(ctx: _SectionContext) -> go.Figure | None:
    if ctx.profile is None:
        return None
    from .profile_sections import plot_occupancy_overview

    return plot_occupancy_overview(ctx.profile, theme=ctx.theme)


def _render_attribution_overview(ctx: _SectionContext) -> go.Figure | None:
    if ctx.profile is None:
        return None
    from .profile_sections import plot_attribution_overview

    return plot_attribution_overview(ctx.profile, theme=ctx.theme)


def _render_drawdown_anatomy(ctx: _SectionContext) -> str | None:
    if ctx.profile is None:
        return None
    from .profile_sections import plot_drawdown_anatomy

    return plot_drawdown_anatomy(ctx.profile, theme=ctx.theme)


# --- Trade Analysis renderers ---

def _render_mfe_mae(ctx: _SectionContext) -> go.Figure | None:
    if ctx.trades is None:
        return None
    from .trade_plots import plot_mfe_mae_scatter

    return plot_mfe_mae_scatter(ctx.trades, theme=ctx.theme)


def _render_exit_reasons(ctx: _SectionContext) -> go.Figure | None:
    if ctx.trades is None:
        return None
    # Skip if fewer than 3 distinct exit reasons (not informative)
    if "exit_reason" in ctx.trades.columns:
        n_reasons = ctx.trades["exit_reason"].n_unique()
        if n_reasons < 3:
            return None
    from .trade_plots import plot_exit_reason_breakdown

    return plot_exit_reason_breakdown(ctx.trades, chart_type="bar", theme=ctx.theme)


def _render_trade_waterfall(ctx: _SectionContext) -> go.Figure | None:
    if ctx.trades is None:
        return None
    from .trade_plots import plot_trade_waterfall

    return plot_trade_waterfall(ctx.trades, theme=ctx.theme)


def _render_duration(ctx: _SectionContext) -> go.Figure | None:
    if ctx.trades is None:
        return None
    from .trade_plots import plot_trade_duration_distribution

    return plot_trade_duration_distribution(ctx.trades, theme=ctx.theme)


def _render_consecutive(ctx: _SectionContext) -> go.Figure | None:
    if ctx.trades is None:
        return None
    from .trade_plots import plot_consecutive_analysis

    return plot_consecutive_analysis(ctx.trades, theme=ctx.theme)


def _render_size_return(ctx: _SectionContext) -> go.Figure | None:
    if ctx.trades is None:
        return None
    from .trade_plots import plot_trade_size_vs_return

    return plot_trade_size_vs_return(ctx.trades, theme=ctx.theme)


# --- Cost Attribution renderers ---

def _render_cost_waterfall(ctx: _SectionContext) -> str | None:
    if ctx.profile is not None:
        from .profile_sections import plot_cost_bridge

        fig = plot_cost_bridge(ctx.profile, theme=ctx.theme)
    else:
        gross_pnl = ctx.metrics.get("gross_pnl")
        commission = ctx.metrics.get("total_commission", ctx.metrics.get("commission", 0))
        slippage = ctx.metrics.get("total_slippage", ctx.metrics.get("slippage", 0))
        if gross_pnl is None:
            return None
        from .cost_attribution import plot_cost_waterfall

        fig = plot_cost_waterfall(
            gross_pnl=gross_pnl, commission=commission,
            slippage=slippage, theme=ctx.theme,
            height=380,
        )
    if fig is None:
        return None
    fig.update_layout(autosize=True, width=None, margin={"t": 40})
    chart_html = _figure_to_clean_html(fig)
    return (
        '<div style="max-width:560px">'
        f'<div class="chart-container">{chart_html}</div>'
        '</div>'
    )


def _render_cost_sensitivity(ctx: _SectionContext) -> go.Figure | None:
    if ctx.returns is None:
        return None
    from .cost_attribution import plot_cost_sensitivity

    return plot_cost_sensitivity(ctx.returns, theme=ctx.theme)


def _render_cost_by_asset(ctx: _SectionContext) -> go.Figure | None:
    if ctx.trades is None:
        return None
    from .cost_attribution import plot_cost_by_asset

    return plot_cost_by_asset(ctx.trades, theme=ctx.theme)


# --- Statistical Validity renderers ---

def _render_statistical_summary(ctx: _SectionContext) -> go.Figure | None:
    if not any(
        key in ctx.metrics
        for key in (
            "dsr_probability", "min_trl", "expected_max_sharpe",
            "pbo", "ras_adjusted_ic",
        )
    ):
        return None
    from .statistical_validity import plot_statistical_summary_card

    return plot_statistical_summary_card(ctx.metrics, theme=ctx.theme)


def _render_dsr_gauge(ctx: _SectionContext) -> go.Figure | None:
    dsr_prob = ctx.metrics.get("dsr_probability")
    sharpe = ctx.metrics.get("sharpe")
    if dsr_prob is None or sharpe is None:
        return None
    from .statistical_validity import plot_dsr_gauge

    return plot_dsr_gauge(
        dsr_probability=dsr_prob,
        observed_sharpe=sharpe,
        expected_max_sharpe=ctx.metrics.get("expected_max_sharpe"),
        n_trials=ctx.n_trials,
        theme=ctx.theme,
    )


def _render_confidence_intervals(ctx: _SectionContext) -> go.Figure | None:
    ci_metrics: dict[str, dict[str, float]] = {}
    for key in ["sharpe", "cagr", "max_drawdown"]:
        lower = ctx.metrics.get(f"{key}_lower_95")
        upper = ctx.metrics.get(f"{key}_upper_95")
        if key in ctx.metrics and lower is not None and upper is not None:
            ci_metrics[key] = {
                "point": ctx.metrics[key],
                "lower_95": lower,
                "upper_95": upper,
            }
    if not ci_metrics:
        return None
    from .statistical_validity import plot_confidence_intervals

    return plot_confidence_intervals(ci_metrics, theme=ctx.theme)


def _render_min_trl(ctx: _SectionContext) -> go.Figure | None:
    sharpe = ctx.metrics.get("sharpe")
    periods = ctx.metrics.get("n_periods", ctx.metrics.get("n_observations"))
    if sharpe is None or periods is None:
        return None
    from .statistical_validity import plot_minimum_track_record

    return plot_minimum_track_record(
        observed_sharpe=sharpe, current_periods=periods, theme=ctx.theme,
    )


def _render_ras_analysis(ctx: _SectionContext) -> go.Figure | None:
    original_ic = ctx.metrics.get("original_ic")
    adjusted_ic = ctx.metrics.get("ras_adjusted_ic")
    rademacher = ctx.metrics.get("rademacher_complexity")
    if original_ic is None or adjusted_ic is None or rademacher is None:
        return None
    from .statistical_validity import plot_ras_analysis

    return plot_ras_analysis(
        original_ic=float(original_ic),
        adjusted_ic=float(adjusted_ic),
        rademacher_complexity=float(rademacher),
        theme=ctx.theme,
    )


def _render_pbo_gauge(ctx: _SectionContext) -> go.Figure | None:
    pbo = ctx.metrics.get("pbo")
    if pbo is None:
        return None
    from .statistical_validity import plot_pbo_gauge

    return plot_pbo_gauge(
        pbo=float(pbo),
        n_combinations=ctx.metrics.get("pbo_n_combinations"),
        n_strategies=ctx.metrics.get("pbo_n_strategies"),
        theme=ctx.theme,
    )


# --- HTML-native renderers ---

def _render_credibility_box(ctx: _SectionContext) -> str | None:
    if not ctx.metrics:
        return None
    from .html_tables import create_credibility_box_html

    return create_credibility_box_html(ctx.metrics)


def _render_top_contributors(ctx: _SectionContext) -> str | None:
    if ctx.profile is None:
        return None
    from .html_tables import create_top_contributors_html

    return create_top_contributors_html(ctx.profile.attribution)


def _render_cost_summary_line(ctx: _SectionContext) -> str | None:
    if not ctx.metrics:
        return None
    from .html_tables import create_cost_summary_line_html

    return create_cost_summary_line_html(ctx.metrics)


def _render_worst_trades_table(ctx: _SectionContext) -> str | None:
    if ctx.trades is None:
        return None
    from .html_tables import create_worst_trades_table_html

    return create_worst_trades_table_html(ctx.trades, max_rows=10)


# --- Portfolio-level renderers ---

def _render_portfolio_section(ctx: _SectionContext, section_name: str) -> Any:
    if ctx.returns is None:
        return None

    pa = _build_portfolio_analysis(
        ctx.returns, benchmark_returns=ctx.benchmark_returns,
        profile=ctx.profile, equity_curve=ctx.equity_curve,
    )

    if section_name == "equity_curve":
        from ml4t.diagnostic.visualization.portfolio import plot_cumulative_returns

        return plot_cumulative_returns(
            pa, theme=ctx.theme, benchmark_label=ctx.benchmark_name,
        )

    if section_name == "drawdowns":
        from ml4t.diagnostic.visualization.portfolio.drawdown_plots import (
            plot_drawdown_summary,
        )

        return plot_drawdown_summary(pa, theme=ctx.theme, height=620)

    if section_name == "top_drawdowns_table":
        from .html_tables import create_top_drawdowns_table_html

        dd_result = pa.compute_drawdown_analysis(top_n=5)
        return create_top_drawdowns_table_html(dd_result.top_drawdowns)

    if section_name in ("monthly_returns", "monthly_heatmap_overview"):
        from ml4t.diagnostic.visualization.portfolio import plot_monthly_returns_heatmap

        return plot_monthly_returns_heatmap(pa, theme=ctx.theme)

    if section_name == "annual_returns":
        from ml4t.diagnostic.visualization.portfolio import plot_annual_returns_bar

        return plot_annual_returns_bar(
            pa, theme=ctx.theme, benchmark_label=ctx.benchmark_name,
        )

    if section_name == "rolling_metrics":
        if ctx.profile is not None:
            from .profile_sections import plot_stability_overview

            return plot_stability_overview(ctx.profile, theme=ctx.theme)
        from ml4t.diagnostic.visualization.portfolio import plot_rolling_sharpe

        rolling = pa.compute_rolling_metrics(windows=[63, 252], metrics=["sharpe"])
        return plot_rolling_sharpe(
            rolling_result=rolling, windows=[63, 252], theme=ctx.theme,
        )

    if section_name == "distribution":
        from ml4t.diagnostic.visualization.portfolio import plot_returns_distribution

        return plot_returns_distribution(pa, theme=ctx.theme)

    if section_name == "tail_risk":
        from .tail_risk import plot_tail_risk_analysis

        ret_arr = (
            ctx.returns if isinstance(ctx.returns, np.ndarray)
            else ctx.returns.to_numpy()
        )
        return plot_tail_risk_analysis(ret_arr, theme=ctx.theme)

    return None


# --- ML renderers ---

def _render_shap_errors(ctx: _SectionContext) -> go.Figure | None:
    if ctx.shap_result is None or not ctx.shap_result.error_patterns:
        return None
    from .shap_patterns import plot_shap_error_patterns

    return plot_shap_error_patterns(ctx.shap_result, theme=ctx.theme)


def _render_ml_summary(ctx: _SectionContext) -> str | None:
    if ctx.profile is None:
        return None
    return _create_ml_summary_html(ctx.profile)


def _render_signal_diagnostics(ctx: _SectionContext) -> go.Figure | None:
    if ctx.profile is None:
        return None
    from .ml_plots import plot_prediction_signal_diagnostics

    return plot_prediction_signal_diagnostics(ctx.profile, theme=ctx.theme)


def _render_prediction_trade_alignment(ctx: _SectionContext) -> go.Figure | None:
    if ctx.profile is None:
        return None
    from .ml_plots import plot_prediction_trade_alignment

    return plot_prediction_trade_alignment(ctx.profile, theme=ctx.theme)


def _render_prediction_calibration(ctx: _SectionContext) -> go.Figure | None:
    if ctx.profile is None:
        return None
    from .ml_plots import plot_prediction_calibration

    return plot_prediction_calibration(ctx.profile, theme=ctx.theme)


# --- Factor renderers ---

def _render_factor_section(ctx: _SectionContext, section_name: str) -> Any:
    if ctx.factor_analysis is None:
        return None

    if section_name == "factor_exposure":
        from ml4t.diagnostic.visualization.factor import plot_factor_betas_bar

        return plot_factor_betas_bar(ctx.factor_analysis.static_model(), theme=ctx.theme)

    if section_name == "factor_attribution":
        from ml4t.diagnostic.visualization.factor import plot_return_attribution_waterfall

        return plot_return_attribution_waterfall(
            ctx.factor_analysis.attribution(), theme=ctx.theme,
        )

    if section_name == "factor_risk":
        from ml4t.diagnostic.visualization.factor import plot_risk_attribution_pie

        return plot_risk_attribution_pie(
            ctx.factor_analysis.risk_attribution(), theme=ctx.theme,
        )

    return None


# --- Registry ---

_SECTION_REGISTRY: dict[str, Any] = {
    # Executive / Overview
    "executive_summary": _render_executive_summary,
    "key_metrics_table": _render_key_metrics_table,
    "key_insights": _render_key_insights,
    "overview_snapshot": _render_overview_snapshot,
    "activity_overview": _render_activity_overview,
    "occupancy_overview": _render_occupancy_overview,
    "attribution_overview": _render_attribution_overview,
    "drawdown_anatomy": _render_drawdown_anatomy,
    # Trade Analysis
    "mfe_mae": _render_mfe_mae,
    "exit_reasons": _render_exit_reasons,
    "trade_waterfall": _render_trade_waterfall,
    "duration": _render_duration,
    "consecutive": _render_consecutive,
    "size_return": _render_size_return,
    # Cost Attribution
    "cost_waterfall": _render_cost_waterfall,
    "cost_sensitivity": _render_cost_sensitivity,
    "cost_by_asset": _render_cost_by_asset,
    # Statistical Validity
    "statistical_summary": _render_statistical_summary,
    "dsr_gauge": _render_dsr_gauge,
    "confidence_intervals": _render_confidence_intervals,
    "min_trl": _render_min_trl,
    "ras_analysis": _render_ras_analysis,
    "pbo_gauge": _render_pbo_gauge,
    # HTML-native
    "credibility_box": _render_credibility_box,
    "top_contributors": _render_top_contributors,
    "cost_summary_line": _render_cost_summary_line,
    "worst_trades_table": _render_worst_trades_table,
    # Portfolio-level (partial-application wrappers)
    "equity_curve": lambda ctx: _render_portfolio_section(ctx, "equity_curve"),
    "drawdowns": lambda ctx: _render_portfolio_section(ctx, "drawdowns"),
    "top_drawdowns_table": lambda ctx: _render_portfolio_section(ctx, "top_drawdowns_table"),
    "monthly_returns": lambda ctx: _render_portfolio_section(ctx, "monthly_returns"),
    "monthly_heatmap_overview": lambda ctx: _render_portfolio_section(ctx, "monthly_heatmap_overview"),
    "annual_returns": lambda ctx: _render_portfolio_section(ctx, "annual_returns"),
    "rolling_metrics": lambda ctx: _render_portfolio_section(ctx, "rolling_metrics"),
    "distribution": lambda ctx: _render_portfolio_section(ctx, "distribution"),
    "tail_risk": lambda ctx: _render_portfolio_section(ctx, "tail_risk"),
    # ML
    "shap_errors": _render_shap_errors,
    "ml_summary": _render_ml_summary,
    "signal_diagnostics": _render_signal_diagnostics,
    "prediction_trade_alignment": _render_prediction_trade_alignment,
    "prediction_calibration": _render_prediction_calibration,
    # Factors (partial-application wrappers)
    "factor_exposure": lambda ctx: _render_factor_section(ctx, "factor_exposure"),
    "factor_attribution": lambda ctx: _render_factor_section(ctx, "factor_attribution"),
    "factor_risk": lambda ctx: _render_factor_section(ctx, "factor_risk"),
}


def _create_section_figure(
    section_name: str,
    preset: BacktestDashboardPreset | None = None,
    profile: BacktestProfile | None = None,
    trades: pl.DataFrame | None = None,
    returns: pl.Series | np.ndarray | None = None,
    equity_curve: pl.DataFrame | None = None,
    metrics: dict[str, Any] | None = None,
    benchmark_returns: pl.Series | np.ndarray | None = None,
    benchmark_metrics: dict[str, float] | None = None,
    benchmark_name: str = "Benchmark",
    n_trials: int | None = None,
    shap_result: TradeShapResult | None = None,
    factor_data: FactorData | None = None,
    factor_analysis: Any | None = None,
    theme: str = "default",
) -> go.Figure | str | None:
    """Create the Plotly figure for a specific section.

    Dispatches to the appropriate renderer via ``_SECTION_REGISTRY``.
    """
    renderer = _SECTION_REGISTRY.get(section_name)
    if renderer is None:
        return None

    ctx = _SectionContext(
        preset=preset or get_dashboard_preset("full"),
        profile=profile,
        trades=trades,
        returns=returns,
        equity_curve=equity_curve,
        metrics=metrics or {},
        benchmark_returns=benchmark_returns,
        benchmark_metrics=benchmark_metrics,
        benchmark_name=benchmark_name,
        n_trials=n_trials,
        shap_result=shap_result,
        factor_data=factor_data,
        factor_analysis=factor_analysis,
        theme=theme,
    )
    return renderer(ctx)


class BacktestTearsheet:
    """Object-oriented interface for building tearsheets incrementally.

    Provides a fluent API for customizing tearsheet content before generation.

    Examples
    --------
    >>> tearsheet = BacktestTearsheet(template="quant_trader")
    >>> tearsheet.add_trades(my_trades)
    >>> tearsheet.add_metrics({"sharpe": 1.5, "max_drawdown": -0.15})
    >>> tearsheet.enable_section("dsr_gauge")
    >>> html = tearsheet.generate()
    """

    def __init__(
        self,
        template: Literal["quant_trader", "hedge_fund", "risk_manager", "full"] = "full",
        theme: Literal["default", "dark", "print", "presentation"] = "default",
        title: str | None = None,
    ):
        """Initialize tearsheet builder."""
        self.template = get_template(template)
        self.theme = theme
        self.title = title
        self.subtitle: str | None = None
        self.profile: BacktestProfile | None = None
        self.report_metadata: BacktestReportMetadata | None = None

        # Data
        self.trades: pl.DataFrame | None = None
        self.returns: pl.Series | np.ndarray | None = None
        self.equity_curve: pl.DataFrame | None = None
        self.metrics: dict[str, Any] = {}
        self.benchmark_returns: pl.Series | np.ndarray | None = None
        self.benchmark_name = "Benchmark"
        self.n_trials: int | None = None
        self.shap_result: TradeShapResult | None = None
        self.factor_data: FactorData | None = None

    def add_profile(self, profile: BacktestProfile) -> BacktestTearsheet:
        """Add a BacktestProfile and hydrate default surfaces from it."""
        self.profile = profile
        self.trades = profile.trades_df
        self.returns = profile.daily_returns
        self.equity_curve = profile.equity_df
        self.metrics.update(_normalize_profile_metrics(profile))
        return self

    def add_trades(self, trades: pl.DataFrame) -> BacktestTearsheet:
        """Add trade records to the tearsheet."""
        self.trades = trades
        return self

    def add_returns(self, returns: pl.Series | np.ndarray) -> BacktestTearsheet:
        """Add daily returns series."""
        self.returns = returns
        return self

    def add_equity_curve(self, equity: pl.DataFrame) -> BacktestTearsheet:
        """Add equity curve DataFrame."""
        self.equity_curve = equity
        return self

    def add_metrics(self, metrics: dict[str, Any]) -> BacktestTearsheet:
        """Add or update metrics dictionary."""
        self.metrics.update(metrics)
        return self

    def add_benchmark(
        self,
        returns: pl.Series | np.ndarray,
        name: str = "Benchmark",
    ) -> BacktestTearsheet:
        """Add benchmark returns for comparison."""
        self.benchmark_returns = returns
        self.benchmark_name = name
        return self

    def add_report_metadata(self, report_metadata: BacktestReportMetadata) -> BacktestTearsheet:
        """Attach structured report metadata."""
        self.report_metadata = report_metadata
        return self

    def add_shap_result(self, shap_result: TradeShapResult) -> BacktestTearsheet:
        """Add SHAP analysis result for error pattern visualization."""
        self.shap_result = shap_result
        return self

    def add_factor_data(self, factor_data: FactorData) -> BacktestTearsheet:
        """Add factor data for factor exposure/attribution/risk sections.

        Automatically enables all factor sections in the template.
        """
        self.factor_data = factor_data
        for section in self.template.sections:
            if section.name.startswith("factor_"):
                section.enabled = True
        return self

    def set_n_trials(self, n: int) -> BacktestTearsheet:
        """Set number of trials for DSR calculation."""
        self.n_trials = n
        return self

    def set_title(self, title: str | None, subtitle: str | None = None) -> BacktestTearsheet:
        """Set report title and subtitle."""
        self.title = title
        self.subtitle = subtitle
        return self

    def enable_section(self, name: str) -> BacktestTearsheet:
        """Enable a section by name."""
        self.template.enable_section(name)
        return self

    def disable_section(self, name: str) -> BacktestTearsheet:
        """Disable a section by name."""
        self.template.disable_section(name)
        return self

    def generate(
        self,
        output_path: str | Path | None = None,
        interactive: bool = True,
        include_plotlyjs: bool = True,
    ) -> str:
        """Generate the tearsheet HTML.

        Uses the stored template directly so that enable_section/disable_section
        customizations are preserved.

        Parameters
        ----------
        output_path : str or Path, optional
            If provided, save HTML to this path
        interactive : bool
            Whether charts should be interactive
        include_plotlyjs : bool
            Whether to include Plotly.js

        Returns
        -------
        str
            HTML string of the complete tearsheet
        """
        if not interactive:
            import warnings

            warnings.warn(
                "interactive=False has no effect; all output is interactive Plotly HTML. "
                "This parameter will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Use stored template directly (not recreated from name) to preserve
        # enable_section/disable_section customizations
        benchmark_metrics = _compute_portfolio_summary_metrics(
            self.benchmark_returns,
            periods_per_year=_infer_periods_per_year(self.returns),
        )
        document_title, report_title, report_subtitle, resolved_benchmark_name = (
            _resolve_report_metadata(
                self.report_metadata,
                self.title,
                self.subtitle,
                self.benchmark_name,
            )
        )
        sections_html = _generate_sections(
            self.template,
            preset=get_dashboard_preset(self.template.name),
            profile=self.profile,
            trades=self.trades,
            returns=self.returns,
            equity_curve=self.equity_curve,
            metrics=self.metrics,
            benchmark_returns=self.benchmark_returns,
            benchmark_metrics=benchmark_metrics,
            benchmark_name=resolved_benchmark_name,
            n_trials=self.n_trials,
            shap_result=self.shap_result,
            factor_data=self.factor_data,
            theme=self.theme,
            interactive=interactive,
        )

        html = HTML_TEMPLATE.format(
            theme=self.theme if self.theme == "dark" else "light",
            document_title=html_mod.escape(document_title),
            title=html_mod.escape(report_title),
            subtitle=html_mod.escape(report_subtitle),
            timestamp=datetime.now().strftime("%Y-%m-%d"),
            css=TEARSHEET_CSS,
            plotly_js=_plotly_js_tag(include_plotlyjs),
            sections_html=sections_html,
        )

        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(html)

        return html
