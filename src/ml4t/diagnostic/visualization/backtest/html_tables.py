"""Dense HTML tables and summary elements for backtest tearsheets.

Provides HTML-native components that don't need Plotly:
- Credibility box (DSR + MinTRL verdicts)
- Top drawdowns table
- Worst trades table
- Cost summary line
"""

from __future__ import annotations

import html as html_mod
from typing import TYPE_CHECKING, Any

import numpy as np

from .executive_summary import (
    DEFAULT_THRESHOLDS,
    TRAFFIC_LIGHT_COLORS,
    get_traffic_light_color,
)

if TYPE_CHECKING:
    import polars as pl

    from ml4t.diagnostic.evaluation.portfolio_analysis.results import DrawdownPeriod


def create_credibility_box_html(
    metrics: dict[str, Any],
    *,
    thresholds: dict[str, dict[str, Any]] | None = None,
) -> str | None:
    """Create a compact credibility strip with traffic-light verdicts.

    Shows DSR probability, MinTRL assessment, and an overall risk sentence.
    Returns None if no credibility metrics are available.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    dsr_prob = metrics.get("dsr_probability")
    min_trl = metrics.get("min_trl")
    n_periods = metrics.get("n_periods", metrics.get("n_observations"))

    items: list[str] = []

    # DSR probability
    if dsr_prob is not None:
        try:
            dsr_val = float(dsr_prob)
        except (TypeError, ValueError):
            dsr_val = None
        if dsr_val is not None and not np.isnan(dsr_val):
            color = get_traffic_light_color(dsr_val, "dsr_probability", thresholds)
            hex_color = TRAFFIC_LIGHT_COLORS.get(color, TRAFFIC_LIGHT_COLORS["neutral"])
            items.append(
                f'<div class="credibility-item">'
                f'<span class="credibility-dot" style="background:{hex_color}"></span>'
                f'<div><div class="credibility-label">Deflated Sharpe</div>'
                f'<div class="credibility-value">'
                f'{dsr_val:.1%}</div></div></div>'
            )

    # MinTRL
    if min_trl is not None and n_periods is not None:
        try:
            trl_val = float(min_trl)
            periods_val = float(n_periods)
        except (TypeError, ValueError):
            trl_val = None
            periods_val = None
        if trl_val is not None and periods_val is not None:
            sufficient = periods_val >= trl_val
            color = "green" if sufficient else "red"
            hex_color = TRAFFIC_LIGHT_COLORS.get(color, TRAFFIC_LIGHT_COLORS["neutral"])
            items.append(
                f'<div class="credibility-item">'
                f'<span class="credibility-dot" style="background:{hex_color}"></span>'
                f'<div><div class="credibility-label">Min Track Record</div>'
                f'<div class="credibility-value">'
                f'{trl_val:.0f} days required ({periods_val:.0f} observed)</div></div></div>'
            )

    # Track record length — show when available
    if n_periods is not None and min_trl is None:
        try:
            periods_val = float(n_periods)
        except (TypeError, ValueError):
            periods_val = None
        if periods_val is not None:
            years = periods_val / 252
            color = "green" if years >= 3 else ("yellow" if years >= 1 else "red")
            hex_color = TRAFFIC_LIGHT_COLORS.get(color, TRAFFIC_LIGHT_COLORS["neutral"])
            verdict = f"{years:.1f} years" if years >= 1 else f"{periods_val:.0f} days"
            items.append(
                f'<div class="credibility-item">'
                f'<span class="credibility-dot" style="background:{hex_color}"></span>'
                f'<div><div class="credibility-label">Track Record</div>'
                f'<div class="credibility-value">{verdict}</div></div></div>'
            )

    if not items:
        return None

    return f'<div class="credibility-box">{"".join(items)}</div>'


def create_top_drawdowns_table_html(
    drawdown_periods: list[DrawdownPeriod],
    *,
    max_rows: int = 5,
) -> str | None:
    """Create a dense HTML table of top drawdown periods.

    Returns None if no drawdown periods are available.
    """
    if not drawdown_periods:
        return None

    periods = drawdown_periods[:max_rows]
    rows: list[str] = []
    for i, dd in enumerate(periods, 1):
        peak = _fmt_date(dd.peak_date)
        valley = _fmt_date(dd.valley_date)
        recovery = _fmt_date(dd.recovery_date) if dd.recovery_date else "Active"
        depth = f"{dd.depth:.1%}" if dd.depth is not None else "N/A"
        duration = f"{dd.duration_days}d" if dd.duration_days is not None else "N/A"
        recovery_days = (
            f"{dd.recovery_days}d" if dd.recovery_days is not None else "—"
        )
        status = "Recovered" if dd.recovery_date else "Active"

        rows.append(
            f"<tr>"
            f"<td>{i}</td>"
            f"<td>{html_mod.escape(peak)}</td>"
            f"<td>{html_mod.escape(valley)}</td>"
            f"<td>{html_mod.escape(recovery)}</td>"
            f"<td>{html_mod.escape(depth)}</td>"
            f"<td>{html_mod.escape(duration)}</td>"
            f"<td>{html_mod.escape(recovery_days)}</td>"
            f"<td>{html_mod.escape(status)}</td>"
            f"</tr>"
        )

    return (
        '<div class="metrics-table-wrap">'
        '<table class="metrics-table">'
        "<thead><tr>"
        "<th>#</th><th>Peak</th><th>Trough</th><th>Recovery</th>"
        "<th>Depth</th><th>Duration</th><th>Recovery</th><th>Status</th>"
        "</tr></thead>"
        f'<tbody>{"".join(rows)}</tbody>'
        "</table></div>"
    )


def create_worst_trades_table_html(
    trades: pl.DataFrame | None,
    *,
    max_rows: int = 10,
) -> str | None:
    """Create a dense HTML table of worst trades sorted by PnL.

    Handles missing columns gracefully. Returns None if trades is empty.
    """

    if trades is None or trades.is_empty():
        return None

    # Find PnL column
    pnl_col = _first_col(trades, ("pnl", "net_pnl", "gross_pnl", "realized_pnl"))
    if pnl_col is None:
        return None

    worst = trades.sort(pnl_col).head(max_rows)

    entry_col = _first_col(worst, ("entry_time", "entry_timestamp", "entry_date"))
    exit_col = _first_col(worst, ("exit_time", "exit_timestamp", "exit_date"))
    sym_col = _first_col(worst, ("symbol", "asset", "ticker"))
    dir_col = _first_col(worst, ("direction", "side"))
    ret_col = _first_col(worst, ("pnl_pct", "return_pct", "return"))
    dur_col = _first_col(worst, ("bars_held", "duration", "holding_period"))
    exit_reason_col = _first_col(worst, ("exit_reason", "close_reason"))

    rows: list[str] = []
    for i in range(worst.height):
        rank = str(i + 1)
        entry = _fmt_cell(worst, i, entry_col, _fmt_date)
        exit_t = _fmt_cell(worst, i, exit_col, _fmt_date)
        symbol = _fmt_cell(worst, i, sym_col)
        direction = _fmt_cell(worst, i, dir_col)
        pnl = _fmt_cell(worst, i, pnl_col, lambda v: f"${v:,.0f}" if v else "N/A")
        ret = _fmt_cell(worst, i, ret_col, lambda v: f"{v:.1%}" if v else "N/A")
        dur = _fmt_cell(worst, i, dur_col, lambda v: str(int(v)) if v else "N/A")
        reason = _fmt_cell(worst, i, exit_reason_col)

        rows.append(
            f"<tr>"
            f"<td>{rank}</td>"
            f"<td>{html_mod.escape(entry)}</td>"
            f"<td>{html_mod.escape(exit_t)}</td>"
            f"<td>{html_mod.escape(symbol)}</td>"
            f"<td>{html_mod.escape(direction)}</td>"
            f"<td>{html_mod.escape(pnl)}</td>"
            f"<td>{html_mod.escape(ret)}</td>"
            f"<td>{html_mod.escape(dur)}</td>"
            f"<td>{html_mod.escape(reason)}</td>"
            f"</tr>"
        )

    return (
        '<div class="metrics-table-wrap">'
        '<table class="metrics-table">'
        "<thead><tr>"
        "<th>#</th><th>Entry</th><th>Exit</th><th>Symbol</th>"
        "<th>Dir</th><th>PnL</th><th>Return</th><th>Dur</th><th>Exit Reason</th>"
        "</tr></thead>"
        f'<tbody>{"".join(rows)}</tbody>'
        "</table></div>"
    )


def create_cost_summary_line_html(metrics: dict[str, Any]) -> str | None:
    """Create a one-line cost summary strip.

    Shows cost drag and total cost (no total return — already in KPI strip).
    Returns None if insufficient cost data.
    """
    total_cost = metrics.get("total_implementation_cost", 0.0)
    gross_pnl = metrics.get("gross_pnl")
    total_pnl = metrics.get("total_pnl")

    items: list[str] = []

    if gross_pnl is not None and total_pnl is not None:
        try:
            gp = float(gross_pnl)
            tp = float(total_pnl)
            drag = gp - tp
            if abs(gp) > 0:
                drag_pct = drag / abs(gp)
                items.append(
                    f'<span class="cost-summary-item">'
                    f'<span class="cost-summary-label">Cost Drag</span>'
                    f'<span class="cost-summary-value">{drag_pct:.1%} of gross</span>'
                    f'</span>'
                )
        except (TypeError, ValueError):
            pass

    if total_cost:
        try:
            tc = float(total_cost)
            items.append(
                f'<span class="cost-summary-item">'
                f'<span class="cost-summary-label">Total Cost</span>'
                f'<span class="cost-summary-value">${tc:,.0f}</span></span>'
            )
        except (TypeError, ValueError):
            pass

    n_trades = metrics.get("num_trades", metrics.get("n_trades"))
    if n_trades is not None:
        items.append(
            f'<span class="cost-summary-item">'
            f'<span class="cost-summary-label">Trades</span>'
            f'<span class="cost-summary-value">{int(n_trades):,}</span></span>'
        )

    if not items:
        return None

    return f'<div class="cost-summary-line">{"".join(items)}</div>'


def create_top_contributors_html(
    attribution: dict[str, Any] | None,
    *,
    top_n: int = 3,
) -> str | None:
    """Create a compact strip showing top contributors and detractors.

    Parameters
    ----------
    attribution : dict or None
        Attribution dict with ``"by_symbol"`` key containing a DataFrame
        with ``symbol`` and ``pnl_contribution_share`` (or ``net_pnl``) columns.
    top_n : int
        Number of contributors/detractors to show.

    Returns None if data is unavailable.
    """
    if attribution is None:
        return None
    by_symbol = attribution.get("by_symbol")
    if by_symbol is None or by_symbol.is_empty():
        return None


    pnl_col = None
    for candidate in ("pnl_contribution_share", "net_pnl", "pnl"):
        if candidate in by_symbol.columns:
            pnl_col = candidate
            break
    if pnl_col is None:
        return None

    sym_col = None
    for candidate in ("symbol", "asset", "ticker"):
        if candidate in by_symbol.columns:
            sym_col = candidate
            break
    if sym_col is None:
        return None

    sorted_df = by_symbol.sort(pnl_col, descending=True)
    top = sorted_df.head(top_n)
    bottom = sorted_df.tail(top_n).sort(pnl_col)

    is_share = pnl_col == "pnl_contribution_share"

    def _fmt(val: float) -> str:
        if is_share:
            return f"{val:+.1%}"
        return f"${val:+,.0f}"

    items: list[str] = []
    # Top contributors
    for i in range(min(top_n, top.height)):
        sym = str(top[sym_col][i])
        val = float(top[pnl_col][i])
        color = TRAFFIC_LIGHT_COLORS["green"] if val >= 0 else TRAFFIC_LIGHT_COLORS["red"]
        items.append(
            f'<span style="color:{color};font-weight:600">'
            f'{html_mod.escape(sym)} {_fmt(val)}</span>'
        )

    items.append('<span style="color:var(--c-text-muted);margin:0 6px">|</span>')

    # Bottom detractors
    for i in range(min(top_n, bottom.height)):
        sym = str(bottom[sym_col][i])
        val = float(bottom[pnl_col][i])
        color = TRAFFIC_LIGHT_COLORS["green"] if val >= 0 else TRAFFIC_LIGHT_COLORS["red"]
        items.append(
            f'<span style="color:{color};font-weight:600">'
            f'{html_mod.escape(sym)} {_fmt(val)}</span>'
        )

    return (
        '<div class="cost-summary-line" style="gap:12px;font-size:13px">'
        '<span style="color:var(--c-text-muted);font-weight:600;'
        'text-transform:uppercase;font-size:10px;letter-spacing:0.06em">'
        'Top / Bottom</span>'
        f'{"".join(items)}'
        '</div>'
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_date(value: Any) -> str:
    if value is None:
        return "N/A"
    s = str(value)
    # Trim time component if present and midnight
    if "T00:00:00" in s:
        return s.split("T")[0]
    if " 00:00:00" in s:
        return s.split(" ")[0]
    return s[:19] if len(s) > 19 else s


def _first_col(df: Any, candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _fmt_cell(
    df: Any,
    row: int,
    col: str | None,
    formatter: Any = None,
) -> str:
    if col is None:
        return "—"
    val = df[col][row]
    if val is None:
        return "—"
    if formatter is not None:
        try:
            return formatter(val)
        except (TypeError, ValueError):
            return str(val)
    return str(val)
