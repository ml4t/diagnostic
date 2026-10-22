"""Compose familiar capacity tables and scenarios from existing analytical kernels."""

from __future__ import annotations

from html import escape
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ml4t.diagnostic.evaluation.portfolio_analysis.inputs import positive_integer

if TYPE_CHECKING:
    import plotly.graph_objects as go


def _limit(value: float, name: str, *, positive: bool, fraction: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value < 0
        or positive
        and value == 0
        or fraction
        and value > 1
    ):
        raise ValueError(
            f"{name} must be finite, {'positive' if positive else 'nonnegative'}"
            + (" and at most one" if fraction else "")
        )
    return float(value)


def _table(frame: pd.DataFrame, name: str) -> str:
    def number(value: float) -> str:
        return "Unbounded" if np.isposinf(value) else f"{value:.6g}"

    return (
        '<div class="capacity-scroll">'
        + frame.to_html(
            table_id="capacity-" + name,
            classes="capacity-table",
            escape=True,
            na_rep="Unavailable",
            float_format=number,
            border=0,
        )
        + "</div>"
    )


def _prepare_capacity_tear_sheet(
    returns: pd.Series,
    positions: pd.DataFrame,
    transactions: pd.DataFrame,
    market_data: pd.DataFrame,
    liquidation_daily_vol_limit: float = 0.2,
    trade_daily_vol_limit: float = 0.05,
    last_n_days: int = 126,
    days_to_liquidate_limit: float = 1,
    estimate_intraday: bool | np.bool_ | str | None = "infer",
    *,
    periods_per_year: int = 252,
    impact: float = 0.1,
    theme: str | None = None,
) -> tuple[go.Figure, list[dict[str, Any]], dict[str, pd.DataFrame]]:
    """Prepare all tables and the figure once for standalone and account reports."""
    from . import capacity, plotting, utils
    from .txn import _dated

    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")
    _dated(returns, "returns")
    if not returns.index.is_unique or not returns.index.is_monotonic_increasing:
        raise ValueError("return dates must be sorted and unique")
    liquidation = _limit(
        liquidation_daily_vol_limit, "liquidation_daily_vol_limit", positive=True, fraction=True
    )
    participation = _limit(
        trade_daily_vol_limit, "trade_daily_vol_limit", positive=False, fraction=True
    )
    threshold = _limit(days_to_liquidate_limit, "days_to_liquidate_limit", positive=False)
    coefficient = _limit(impact, "impact", positive=False)
    recent = positive_integer(last_n_days, "last_n_days")
    scale = positive_integer(periods_per_year, "periods_per_year")
    selected = utils.check_intraday(estimate_intraday, returns, positions, transactions)
    full = capacity.get_max_days_to_liquidate_by_ticker(selected, market_data, liquidation, 1e6, 5)
    recent_positions = capacity.get_max_days_to_liquidate_by_ticker(
        selected, market_data, liquidation, 1e6, 5, recent
    )
    traded = capacity.get_low_liquidity_transactions(transactions, market_data)
    recent_traded = capacity.get_low_liquidity_transactions(transactions, market_data, recent)
    views = (
        ("liquidation-full", "Whole backtest: liquidation", full, "days_to_liquidate", threshold),
        (
            "liquidation-recent",
            f"Recent {recent} civil days: liquidation",
            recent_positions,
            "days_to_liquidate",
            threshold,
        ),
        (
            "participation-full",
            "Whole backtest: transaction participation",
            traded,
            "max_pct_bar_consumed",
            participation * 100,
        ),
        (
            "participation-recent",
            f"Recent {recent} civil days: transaction participation",
            recent_traded,
            "max_pct_bar_consumed",
            participation * 100,
        ),
    )
    first_return = float(returns.iloc[0]) if len(returns) else np.nan
    first_nav = float(selected.iloc[0].sum(min_count=1)) if len(selected) else np.nan
    capital = (
        first_nav / (1 + first_return)
        if np.isfinite(first_return) and first_return > -1
        else np.nan
    )
    if np.isfinite(capital) and capital > 0:
        figure = plotting.plot_capacity_sweep(
            returns,
            transactions,
            market_data,
            capital,
            periods_per_year=scale,
            impact=coefficient,
            theme=theme,
        )
    else:
        figure = plotting._figure(
            "Capital Base Performance Sweep", "annualized Sharpe ratio", None, {"theme": theme}
        )
        figure.add_annotation(
            text="Unavailable: positive beginning capital cannot be inferred from the first supplied NAV/return.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font_size=14,
        )
    profile = {
        "method": "legacy_pyfolio_capacity_sheet",
        "positions_kind": "estimated_intraday" if selected is not positions else "supplied",
        "position_observations": len(selected),
        "tied_date_observations": len(selected) - selected.index.nunique(),
        "liquidation_participation": liquidation,
        "transaction_participation": participation,
        "days_to_liquidate_limit": threshold,
        "periods_per_year": scale,
        "beginning_capital": float(capital) if np.isfinite(capital) and capital > 0 else None,
    }
    figure.update_layout(meta={**(figure.layout.meta or {}), "capacity_report": profile})
    note = (
        f"Liquidation assumes {liquidation:g} ({liquidation * 100:g}%) daily-bar participation, "
        f"five prior supplied market observations and a fixed 1 million capital base. "
        "All capital and notional values use one supplied account/market currency, without FX conversion. "
        f"Both views flag durations greater than {threshold:g} observations. "
        f"Transaction tables flag participation above {participation * 100:g}%. "
        "Unknown estimates remain Unavailable; known zero-volume exposure is Unbounded. "
        "Recent windows use the helpers' civil-day endpoints, not exchange-session counts. "
        f"Positions: {profile['positions_kind']}; {len(selected)} observations, "
        f"{profile['tied_date_observations']} additional tied-date candidates. "
        "An estimate is retrospective, not actual intraday inventory or continuous-time peak liquidity. "
        f"Hypothetical additional impact uses coefficient {coefficient:g} and compounded closing NAV; "
        "observed costs are not charged twice. No empirical impact calibration or execution prediction is claimed. "
        f"Sharpe is dimensionless and uses {scale} observations per year. "
        "Scroll wide tables horizontally; financial values and labels are not split across lines."
    )
    sections = [
        {
            "title": "Capacity assumptions and coverage",
            "text": "<style>.capacity-scroll{overflow-x:auto;max-width:100%}.capacity-table{font:inherit;font-size:14px;white-space:nowrap}.capacity-table th,.capacity-table td{padding:5px 10px;text-align:right}</style><p>"
            + escape(note)
            + "</p>",
        }
    ]
    tables = {}
    for name, title, frame, column, limit in views:
        flagged = frame[(frame[column] > limit) | frame[column].isna()]
        tables[title] = flagged.copy(deep=True)
        sections.append(
            {
                "title": title,
                "text": f"<p>{len(flagged)} flagged or unavailable assets; {frame[column].notna().sum()} assets have an estimate. Values are in days/observations or percent as labeled.</p>"
                + _table(flagged, name),
            }
        )
    sections.append({"title": "Capital base performance sweep", "figure_index": 0})
    sections[0]["description"] = note
    return figure, sections, tables


def create_capacity_tear_sheet(
    returns: pd.Series,
    positions: pd.DataFrame,
    transactions: pd.DataFrame,
    market_data: pd.DataFrame,
    liquidation_daily_vol_limit: float = 0.2,
    trade_daily_vol_limit: float = 0.05,
    last_n_days: int = 126,
    days_to_liquidate_limit: float = 1,
    estimate_intraday: bool | np.bool_ | str | None = "infer",
    return_fig: bool = False,
    *,
    periods_per_year: int = 252,
    impact: float = 0.1,
    output_file: str | Path | None = None,
    theme: str | None = None,
) -> str | go.Figure:
    """Render complete capacity tables and a sweep through the existing renderer.

    return_fig=True returns the Plotly sweep; output_file still writes every
    table. Legacy fixed1m capital, five prior rows and civil-day lookbacks remain
    explicit. Both views honor the selected threshold, unlike the reference's
    hardcoded recent threshold. Native capacity defaults are unchanged.
    """
    from ml4t.diagnostic.visualization.report_generation import combine_figures_to_html

    if not isinstance(return_fig, bool | np.bool_):
        raise TypeError("return_fig must be a boolean")
    figure, sections, _ = _prepare_capacity_tear_sheet(
        returns,
        positions,
        transactions,
        market_data,
        liquidation_daily_vol_limit,
        trade_daily_vol_limit,
        last_n_days,
        days_to_liquidate_limit,
        estimate_intraday,
        periods_per_year=periods_per_year,
        impact=impact,
        theme=theme,
    )
    html = combine_figures_to_html(
        [figure],
        title="Supplied-market capacity analysis",
        sections=sections,
        output_file=output_file,
        theme=theme,
        include_plotlyjs=True,
    )
    return figure if return_fig else html
