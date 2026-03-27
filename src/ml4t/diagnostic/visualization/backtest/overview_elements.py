"""HTML element renderers for the Overview tab layout.

Provides standalone HTML generators for sidebar metrics, validity cards,
activity strips, and ML summary strips used across tearsheet tabs.
"""

from __future__ import annotations

import html as html_mod
import math
from typing import Any

from ml4t.diagnostic.visualization._colors import FACTOR_COLORS, FACTOR_DESCRIPTIONS

# =============================================================================
# Key Resolution Helpers
# =============================================================================


def _get(metrics: dict[str, Any], *keys: str) -> Any | None:
    """Return the first matching key from *metrics*, or ``None``."""
    for k in keys:
        if k in metrics:
            return metrics[k]
    return None


# =============================================================================
# Formatting Helpers
# =============================================================================


def _fmt_ratio(v: Any) -> str:
    if v is None:
        return "\u2014"
    return f"{float(v):.2f}"


def _fmt_pct(v: Any) -> str:
    """Format a fraction as a percentage (e.g., 0.15 → '15.0%', 1.35 → '135.0%')."""
    if v is None:
        return "\u2014"
    return f"{float(v):.1%}"


def _fmt_int(v: Any) -> str:
    if v is None:
        return "\u2014"
    return f"{int(v):,}"


def _fmt_duration(v: Any) -> str:
    if v is None:
        return "\u2014"
    return f"{float(v):.1f} d"


def _fmt_shape(v: Any) -> str:
    if v is None:
        return "\u2014"
    return f"{float(v):.2f}"


def _fmt_pct_best_worst(v: Any) -> str:
    if v is None:
        return "\u2014"
    return f"{float(v):.1%}"


# =============================================================================
# 1. Metrics Sidebar
# =============================================================================

# Each group: (header, [(label, keys, formatter, negative_is_bad)])
_SIDEBAR_GROUPS: list[tuple[str, list[tuple[str, tuple[str, ...], Any, bool]]]] = [
    (
        "RETURNS",
        [
            ("CAGR", ("cagr",), _fmt_pct, True),
            ("Best Month", ("best_month",), _fmt_pct_best_worst, False),
            ("Worst Month", ("worst_month",), _fmt_pct_best_worst, False),
            ("Omega", ("omega_ratio", "omega"), _fmt_ratio, True),
        ],
    ),
    (
        "RISK-ADJUSTED",
        [
            ("Sharpe", ("sharpe_ratio", "sharpe"), _fmt_ratio, True),
            ("Sortino", ("sortino_ratio",), _fmt_ratio, True),
            ("Calmar", ("calmar_ratio",), _fmt_ratio, True),
            ("Volatility", ("annualized_volatility", "volatility"), _fmt_pct, False),
        ],
    ),
    (
        "BENCHMARK",
        [
            ("Alpha", ("alpha",), _fmt_pct, True),
            ("Beta", ("beta",), _fmt_ratio, False),
            ("Info Ratio", ("information_ratio",), _fmt_ratio, True),
            ("Tracking Err", ("tracking_error",), _fmt_pct, False),
        ],
    ),
    (
        "DRAWDOWN",
        [
            ("Max Drawdown", ("max_drawdown",), _fmt_pct, False),
            ("Avg Drawdown", ("avg_drawdown",), _fmt_pct, False),
            ("Max DD Duration", ("max_drawdown_duration",), _fmt_duration, False),
            ("DD Episodes", ("num_drawdowns", "drawdown_episodes"), _fmt_int, False),
        ],
    ),
    (
        "TRADING",
        [
            ("Trades", ("num_trades", "n_trades"), _fmt_int, False),
            ("Win Rate", ("win_rate",), _fmt_pct, True),
            ("Profit Factor", ("profit_factor",), _fmt_ratio, True),
            ("Avg Holding", ("avg_bars_held",), _fmt_duration, False),
        ],
    ),
    (
        "TAIL & SHAPE",
        [
            ("VaR 95", ("var_95",), _fmt_pct, False),
            ("CVaR 95", ("cvar_95",), _fmt_pct, False),
            ("Skewness", ("skewness",), _fmt_shape, False),
            ("Kurtosis", ("kurtosis",), _fmt_shape, False),
        ],
    ),
]


def create_metrics_sidebar_html(metrics: dict[str, Any]) -> str:
    """Render a grouped metrics sidebar (QuantConnect sidebar pattern).

    Designed to sit alongside the equity chart at ~34 % width.

    Parameters
    ----------
    metrics : dict[str, Any]
        Flat dictionary of metric values. Missing keys are silently skipped.

    Returns
    -------
    str
        HTML fragment for the sidebar.
    """
    groups_html: list[str] = []

    for header, rows in _SIDEBAR_GROUPS:
        row_parts: list[str] = []
        for label, keys, formatter, neg_is_bad in rows:
            value = _get(metrics, *keys)
            if value is None:
                continue
            formatted = formatter(value)
            value_cls = "metrics-sidebar-value"
            if neg_is_bad and isinstance(value, (int, float)) and float(value) < 0:
                value_cls += " negative"
            row_parts.append(
                f'<div class="metrics-sidebar-row">'
                f'<span class="metrics-sidebar-label">'
                f"{html_mod.escape(label)}</span>"
                f'<span class="{value_cls}" '
                f'style="font-variant-numeric: tabular-nums">'
                f"{html_mod.escape(formatted)}</span>"
                f"</div>"
            )
        if not row_parts:
            continue
        groups_html.append(
            f'<div class="metrics-sidebar-group">'
            f'<div class="metrics-sidebar-header">'
            f"{html_mod.escape(header)}</div>"
            f"{''.join(row_parts)}"
            f"</div>"
        )

    return f'<div class="metrics-sidebar">{"".join(groups_html)}</div>'


# =============================================================================
# 2. Statistical Validity Card
# =============================================================================

_GREEN = "#10b981"
_YELLOW = "#eab308"
_RED = "#ef4444"


def _dsr_color(probability: float | None) -> str:
    if probability is None:
        return _RED
    if probability > 0.95:
        return _GREEN
    if probability >= 0.5:
        return _YELLOW
    return _RED


def _mintrl_color(
    observed_sr: float | None,
    min_trl: float | None,
) -> str:
    if observed_sr is None or min_trl is None:
        return _RED
    if min_trl <= 0:
        return _GREEN
    ratio = observed_sr / min_trl if min_trl != 0 else 0.0
    if ratio > 1.0:
        return _GREEN
    if ratio > 0.5:
        return _YELLOW
    return _RED


def _track_color(years: float | None) -> str:
    if years is None:
        return _RED
    if years > 5:
        return _GREEN
    if years >= 2:
        return _YELLOW
    return _RED


def create_validity_card_html(metrics: dict[str, Any]) -> str:
    """Render the 3-column statistical validity card.

    Parameters
    ----------
    metrics : dict[str, Any]
        Flat dictionary of metric values.

    Returns
    -------
    str
        HTML fragment for the validity card.
    """
    # ---- DSR column ----
    dsr_prob = _get(metrics, "dsr_probability")
    observed_sr = _get(metrics, "sharpe_ratio", "sharpe")
    dsr_dot = _dsr_color(dsr_prob)

    dsr_rows: list[str] = []
    if dsr_prob is not None:
        dsr_rows.append(_validity_metric_row("Probability", f"{float(dsr_prob):.1%}"))
    if observed_sr is not None:
        dsr_rows.append(_validity_metric_row("Observed SR", f"{float(observed_sr):.2f}"))
    dsr_context = "Need &gt;95% for confidence"

    # ---- MinTRL column ----
    min_trl = _get(metrics, "min_trl")
    mintrl_dot = _mintrl_color(
        float(observed_sr) if observed_sr is not None else None,
        float(min_trl) if min_trl is not None else None,
    )
    n_periods = _get(metrics, "n_periods", "n_observations")

    mintrl_rows: list[str] = []
    if min_trl is not None:
        trl_f = float(min_trl)
        if not math.isfinite(trl_f) or trl_f > 36500:
            mintrl_rows.append(_validity_metric_row("Min TRL", "—"))
        else:
            mintrl_rows.append(_validity_metric_row("Min TRL", f"{trl_f:.1f}"))
    if observed_sr is not None:
        mintrl_rows.append(_validity_metric_row("Observed SR", f"{float(observed_sr):.2f}"))
    if n_periods is not None:
        mintrl_rows.append(_validity_metric_row("Observations", f"{int(n_periods):,}"))
    mintrl_context = "SR must exceed required SR"

    # ---- Track Record column ----
    years = float(n_periods) / 252.0 if n_periods is not None else None
    track_dot = _track_color(years)
    n_trades = _get(metrics, "num_trades", "n_trades")

    track_rows: list[str] = []
    if years is not None:
        track_rows.append(_validity_metric_row("Track Record", f"{years:.1f} yrs"))
    if n_periods is not None:
        track_rows.append(_validity_metric_row("Trading Days", f"{int(n_periods):,}"))
    if n_trades is not None:
        track_rows.append(_validity_metric_row("Trades", f"{int(n_trades):,}"))
    track_context = "Need &gt;5 yrs for full confidence"

    # ---- Sharpe Significance column ----
    sharpe_pvalue = _get(metrics, "sharpe_pvalue")
    sharpe_se = _get(metrics, "sharpe_se")
    ret_skew = _get(metrics, "return_skewness")
    ret_kurt = _get(metrics, "return_kurtosis")
    ret_ac = _get(metrics, "return_autocorrelation")

    if sharpe_pvalue is not None:
        sig_dot = _GREEN if sharpe_pvalue < 0.05 else (_YELLOW if sharpe_pvalue < 0.10 else _RED)
    else:
        sig_dot = _RED

    sig_rows: list[str] = []
    if sharpe_pvalue is not None:
        sig_rows.append(_validity_metric_row("p-value", f"{float(sharpe_pvalue):.3f}"))
    if sharpe_se is not None:
        sig_rows.append(_validity_metric_row("SR Std Error", f"{float(sharpe_se):.3f}"))
    if ret_skew is not None:
        sig_rows.append(_validity_metric_row("Skewness", f"{float(ret_skew):.2f}"))
    if ret_kurt is not None:
        sig_rows.append(_validity_metric_row("Kurtosis", f"{float(ret_kurt):.2f}"))
    if ret_ac is not None:
        sig_rows.append(_validity_metric_row("Autocorr", f"{float(ret_ac):.3f}"))
    sig_context = "p&lt;0.05 rejects H0: SR=0"

    # ---- Assemble ----
    columns = [
        _validity_column("Deflated Sharpe Ratio", dsr_dot, dsr_rows, dsr_context),
        _validity_column("Min Track Record Length", mintrl_dot, mintrl_rows, mintrl_context),
        _validity_column("Track Record", track_dot, track_rows, track_context),
    ]
    if sig_rows:
        columns.append(_validity_column("Sharpe Significance", sig_dot, sig_rows, sig_context))

    return (
        '<div class="validity-card">'
        '<div class="validity-card-header">STATISTICAL VALIDITY</div>'
        '<div class="validity-card-body">'
        f"{''.join(columns)}"
        "</div>"
        "</div>"
    )


def _validity_metric_row(label: str, value: str) -> str:
    return (
        '<div class="validity-metric">'
        f'<span class="validity-metric-label">{html_mod.escape(label)}</span>'
        f'<span class="validity-metric-value" '
        f'style="font-variant-numeric: tabular-nums">'
        f"{html_mod.escape(value)}</span>"
        "</div>"
    )


def _validity_column(
    title: str,
    dot_color: str,
    rows: list[str],
    context: str,
) -> str:
    return (
        '<div class="validity-column">'
        '<div class="validity-column-header">'
        f'<span class="validity-dot" style="background: {dot_color}"></span>'
        f"{html_mod.escape(title)}"
        "</div>"
        f"{''.join(rows)}"
        f'<div class="validity-context">{context}</div>'
        "</div>"
    )


# =============================================================================
# 3. Activity Strip (Trading tab)
# =============================================================================


def _kpi_card(label: str, value: str) -> str:
    return (
        '<div class="executive-kpi">'
        f'<div class="executive-kpi-label">{html_mod.escape(label)}</div>'
        f'<div class="executive-kpi-value" '
        f'style="font-variant-numeric: tabular-nums">'
        f"{html_mod.escape(value)}</div>"
        "</div>"
    )


def create_activity_strip_html(metrics: dict[str, Any]) -> str:
    """Render a KPI strip with trading activity metrics.

    Parameters
    ----------
    metrics : dict[str, Any]
        Flat dictionary of metric values.

    Returns
    -------
    str
        HTML fragment using ``.executive-strip`` / ``.executive-kpi`` classes.
    """
    cards: list[str] = []

    total_trades = _get(metrics, "num_trades", "n_trades")
    if total_trades is not None:
        cards.append(_kpi_card("TOTAL TRADES", f"{int(total_trades):,}"))

    total_fills = _get(metrics, "num_fills")
    if total_fills is not None:
        cards.append(_kpi_card("TOTAL FILLS", f"{int(total_fills):,}"))

    symbols = _get(metrics, "unique_symbols_traded", "n_symbols")
    if symbols is not None:
        cards.append(_kpi_card("SYMBOLS", f"{int(symbols):,}"))

    avg_holding = _get(metrics, "avg_bars_held")
    if avg_holding is not None:
        cards.append(_kpi_card("AVG HOLDING", f"{float(avg_holding):.1f} days"))

    avg_turnover = _get(metrics, "avg_turnover")
    if avg_turnover is not None:
        cards.append(_kpi_card("AVG TURNOVER", f"{float(avg_turnover):.2f}"))

    time_in_market = _get(metrics, "time_in_market")
    if time_in_market is not None:
        fv = float(time_in_market)
        formatted = f"{fv:.1%}"
        cards.append(_kpi_card("TIME IN MARKET", formatted))

    if not cards:
        return ""

    return f'<div class="executive-strip">{"".join(cards)}</div>'


# =============================================================================
# 5. Factor Legend Sidebar
# =============================================================================

_FACTOR_LEGEND: list[tuple[str, str, str]] = [
    (name, FACTOR_COLORS[name], FACTOR_DESCRIPTIONS.get(name, name))
    for name in ("Mkt-RF", "SMB", "HML", "RMW", "CMA")
]


def create_factor_legend_html(
    factor_names: list[str],
    *,
    betas: dict[str, float] | None = None,
    r_squared: float | None = None,
    source: str = "fama_french",
) -> str:
    """Render a factor legend sidebar with colored dots, descriptions, and source.

    Designed to sit alongside the exposure chart at ~34% width,
    mirroring the Overview metrics sidebar pattern.
    """
    rows: list[str] = []

    for name, color, desc in _FACTOR_LEGEND:
        if name not in factor_names:
            continue
        beta_str = ""
        if betas and name in betas:
            beta_str = (
                f'<span style="font-weight:500;font-size:12px;'
                f'font-variant-numeric:tabular-nums;white-space:nowrap">'
                f"{betas[name]:.3f}</span>"
            )
        rows.append(
            f'<div style="display:flex;align-items:baseline;gap:8px;'
            f'padding:4px 0;line-height:1.5">'
            f'<span style="display:inline-block;width:10px;height:10px;'
            f'border-radius:50%;background:{color};flex-shrink:0;'
            f'margin-top:3px"></span>'
            f'<div style="flex:1;min-width:0">'
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:baseline">'
            f'<span style="font-weight:600;font-size:12px">'
            f"{html_mod.escape(name)}</span>"
            f"{beta_str}"
            f"</div>"
            f'<div style="font-size:10px;color:var(--c-text-muted,#64748b);'
            f'line-height:1.3">{html_mod.escape(desc)}</div>'
            f"</div></div>"
        )

    # Model fit summary
    model_section = ""
    if r_squared is not None:
        model_section = (
            '<div style="margin-top:12px;padding-top:8px;'
            'border-top:1px solid var(--c-border-light,#e2e8f0)">'
            '<div class="metrics-sidebar-header">MODEL FIT</div>'
            '<div class="metrics-sidebar-row">'
            '<span class="metrics-sidebar-label">R\u00b2</span>'
            '<span class="metrics-sidebar-value"'
            ' style="font-variant-numeric:tabular-nums">'
            f"{r_squared:.3f}</span></div>"
            "</div>"
        )

    # Source attribution
    source_key = source.lower().replace(" ", "_")
    if "fama" in source_key or "french" in source_key or source_key.startswith("ff"):
        source_label = "Kenneth French Data Library"
    elif "aqr" in source_key:
        source_label = "AQR Capital Management"
    else:
        source_label = source

    source_html = (
        '<div style="margin-top:16px;padding-top:8px;'
        'border-top:1px solid var(--c-border-light,#e2e8f0)">'
        '<div style="font-size:9px;text-transform:uppercase;'
        'letter-spacing:0.5px;color:var(--c-text-muted,#64748b);'
        'margin-bottom:2px">Data Source</div>'
        f'<div style="font-size:11px;color:var(--c-text-secondary,#475569)">'
        f"{html_mod.escape(source_label)}</div>"
        "</div>"
    )

    return (
        f'<div class="metrics-sidebar" style="padding:16px">'
        f'<div class="metrics-sidebar-header">FACTORS</div>'
        f"{''.join(rows)}"
        f"{model_section}"
        f"{source_html}"
        f"</div>"
    )

# =============================================================================
# 4. ML Summary Strip (ML tab)
# =============================================================================


def create_ml_summary_strip_html(metrics: dict[str, Any]) -> str:
    """Render a KPI strip with ML signal quality metrics.

    Parameters
    ----------
    metrics : dict[str, Any]
        Flat dictionary of metric values.

    Returns
    -------
    str
        HTML fragment using ``.executive-strip`` / ``.executive-kpi`` classes.
    """
    cards: list[str] = []

    mean_ic = _get(metrics, "mean_ic")
    if mean_ic is not None:
        cards.append(_kpi_card("MEAN IC", f"{float(mean_ic):.4f}"))

    ic_tstat = _get(metrics, "ic_tstat")
    if ic_tstat is not None:
        cards.append(_kpi_card("IC T-STAT", f"{float(ic_tstat):.2f}"))

    hit_rate = _get(metrics, "hit_rate")
    if hit_rate is not None:
        fv = float(hit_rate)
        formatted = f"{fv:.1%}"
        cards.append(_kpi_card("HIT RATE", formatted))

    coverage = _get(metrics, "coverage")
    if coverage is not None:
        fv = float(coverage)
        formatted = f"{fv:.1%}"
        cards.append(_kpi_card("COVERAGE", formatted))

    selection_rate = _get(metrics, "selection_rate")
    if selection_rate is not None:
        fv = float(selection_rate)
        formatted = f"{fv:.1%}"
        cards.append(_kpi_card("SELECTION RATE", formatted))

    if not cards:
        return ""

    return f'<div class="executive-strip">{"".join(cards)}</div>'
