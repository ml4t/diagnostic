"""ML-oriented backtest plots built from prediction surfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from ml4t.diagnostic.visualization._colors import COLORS
from ml4t.diagnostic.visualization.core import get_theme_config, validate_theme

if TYPE_CHECKING:
    from ml4t.diagnostic.integration.backtest_profile import BacktestProfile


def plot_prediction_signal_diagnostics(
    profile: BacktestProfile,
    *,
    theme: Literal["default", "dark", "print", "presentation"] = "default",
) -> go.Figure | None:
    """Plot IC, quantile, and regime diagnostics from the raw prediction surface."""
    validate_theme(theme)
    predictions = profile.predictions_df
    if predictions.is_empty():
        return None

    date_col = "timestamp" if "timestamp" in predictions.columns else None
    asset_col = "asset" if "asset" in predictions.columns else None
    score_col = "prediction_value" if "prediction_value" in predictions.columns else None
    outcome_col = _first_present_column(predictions, ("y_true", "actual", "target", "realized_return", "forward_return"))
    if date_col is None or asset_col is None or score_col is None or outcome_col is None:
        return None

    frame = (
        predictions.select([date_col, asset_col, score_col, outcome_col])
        .rename(
            {
                date_col: "date",
                asset_col: "asset",
                score_col: "score",
                outcome_col: "outcome",
            }
        )
        .with_columns(pl.col("date").cast(pl.Date, strict=False))
        .filter(
            pl.col("date").is_not_null()
            & pl.col("score").is_not_null()
            & pl.col("outcome").is_not_null()
        )
    )
    if frame.is_empty():
        return None

    daily_ic = _compute_daily_ic(frame)
    if daily_ic.is_empty():
        return None
    quantile_summary = _compute_quantile_summary(frame, n_quantiles=10)
    regime_summary = _compute_regime_summary(profile, daily_ic)

    # Compute rolling returns for IC vs Returns overlay (row 3)
    equity = profile.equity_df
    ic_with_ret = daily_ic.with_columns(pl.lit(None).cast(pl.Float64).alias("rolling_return_21"))
    if not equity.is_empty() and "timestamp" in equity.columns:
        eq = equity.select(pl.col("timestamp").cast(pl.Date, strict=False).alias("date"))
        if "return" in equity.columns:
            eq = eq.with_columns(equity["return"].alias("daily_ret"))
        elif "cumulative_return" in equity.columns:
            cum = equity["cumulative_return"]
            eq = eq.with_columns(
                ((1 + cum) / (1 + cum.shift(1)) - 1).alias("daily_ret")
            )
        elif "equity" in equity.columns:
            eq_val = equity["equity"]
            eq = eq.with_columns(
                (eq_val / eq_val.shift(1) - 1).alias("daily_ret")
            )
        if "daily_ret" in eq.columns:
            eq = eq.with_columns(
                pl.col("daily_ret")
                .rolling_mean(window_size=21, min_samples=5)
                .alias("rolling_return_21")
            )
            ic_with_ret = daily_ic.join(eq.select("date", "rolling_return_21"), on="date", how="left")

    theme_config = get_theme_config(theme)

    # 4-row layout: Daily IC, Rolling IC vs Returns, Decile Returns, Regime
    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=(
            "Daily Information Coefficient",
            "Rolling IC vs Rolling Returns (21d)",
            "Realized Return by Prediction Decile",
            "IC by Strategy Regime",
        ),
        vertical_spacing=0.08,
        row_heights=[0.30, 0.28, 0.22, 0.20],
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": True}],
            [{}],
            [{}],
        ],
    )

    # Row 1: Daily IC + rolling mean (no legend — title + colors are self-documenting)
    ic_color = theme_config["colorway"][0]
    fig.add_trace(
        go.Scatter(
            x=daily_ic["date"].to_list(),
            y=daily_ic["ic"].to_list(),
            mode="lines", name="Daily IC",
            line={"width": 0.8, "color": "rgba(10,22,40,0.25)"},
            showlegend=False,
        ),
        row=1, col=1,
    )
    if "rolling_ic_21" in daily_ic.columns:
        fig.add_trace(
            go.Scatter(
                x=daily_ic["date"].to_list(),
                y=daily_ic["rolling_ic_21"].to_list(),
                mode="lines", name="21d Mean",
                line={"width": 2, "color": ic_color},
                showlegend=False,
            ),
            row=1, col=1,
        )

    # Row 2: Rolling IC (left y) vs Rolling Returns (right y) — dual axis
    ret_color = COLORS["warning"]
    if "rolling_ic_21" in ic_with_ret.columns:
        fig.add_trace(
            go.Scatter(
                x=ic_with_ret["date"].to_list(),
                y=ic_with_ret["rolling_ic_21"].to_list(),
                mode="lines", name="Rolling IC",
                line={"width": 1.5, "color": ic_color},
                hovertemplate="IC: %{y:.3f}<extra></extra>",
                showlegend=False,
            ),
            row=2, col=1, secondary_y=False,
        )
    if "rolling_return_21" in ic_with_ret.columns:
        rr = ic_with_ret["rolling_return_21"]
        if rr.drop_nulls().len() > 0:
            fig.add_trace(
                go.Scatter(
                    x=ic_with_ret["date"].to_list(),
                    y=rr.to_list(),
                    mode="lines", name="Rolling Return",
                    line={"width": 1.5, "color": ret_color},
                    hovertemplate="Return: %{y:.2%}<extra></extra>",
                    showlegend=False,
                ),
                row=2, col=1, secondary_y=True,
            )

    # Row 3: Decile returns
    if not quantile_summary.is_empty():
        q_labels = quantile_summary["quantile_label"].to_list()
        q_means = quantile_summary["mean_outcome"].to_list()
        q_counts = (
            quantile_summary["count"].to_list()
            if "count" in quantile_summary.columns
            else [None] * len(q_labels)
        )
        bar_colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in q_means]
        text_labels = [f"N={c:,}" if c is not None else "" for c in q_counts]
        fig.add_trace(
            go.Bar(
                x=q_labels, y=q_means, name="Mean Return",
                marker_color=bar_colors,
                text=text_labels, textposition="outside", textfont={"size": 9},
                showlegend=False,
            ),
            row=3, col=1,
        )

    # Row 4: IC by regime — auto-scale y to data range
    if not regime_summary.is_empty():
        regime_ics = regime_summary["mean_ic"].to_list()
        r_colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in regime_ics]
        fig.add_trace(
            go.Bar(
                x=regime_summary["regime"].to_list(),
                y=regime_ics,
                marker_color=r_colors, showlegend=False,
                hovertemplate="%{x}: IC = %{y:.4f}<extra></extra>",
            ),
            row=4, col=1,
        )
        # Tight y-axis around the data with ~20% padding
        if regime_ics:
            y_min = min(regime_ics)
            y_max = max(regime_ics)
            pad = max(abs(y_min), abs(y_max)) * 0.3 + 0.001
            fig.update_yaxes(range=[y_min - pad, y_max + pad], row=4, col=1)

    fig.update_layout(
        title="Prediction Diagnostics",
        height=1100,
        template=theme_config.get("template", "plotly_white"),
        paper_bgcolor=theme_config.get("paper_bgcolor"),
        plot_bgcolor=theme_config.get("plot_bgcolor"),
        font={"color": theme_config.get("font_color")},
        margin={"l": 48, "r": 48, "t": 72, "b": 48},
        showlegend=False,
    )
    fig.update_yaxes(title_text="IC", row=1, col=1)
    fig.update_yaxes(title_text="Rolling IC", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Return", tickformat=".1%", row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Mean Return", tickformat=".2%", row=3, col=1)
    fig.update_yaxes(title_text="Mean IC", row=4, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"], line_width=0.5, row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"], line_width=0.5, row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"], line_width=0.5, row=4, col=1)
    return fig


def plot_ic_time_series(
    profile: BacktestProfile,
    *,
    theme: Literal["default", "dark", "print", "presentation"] = "default",
) -> go.Figure | None:
    """Plot daily IC with rolling mean — compact single-chart version."""
    validate_theme(theme)
    predictions = profile.predictions_df
    if predictions.is_empty():
        return None

    date_col = _first_present_column(predictions, ("timestamp", "date", "session_date"))
    asset_col = _first_present_column(predictions, ("asset",))
    score_col = _first_present_column(
        predictions,
        ("prediction_value", "score", "prediction", "y_pred", "y_score", "ml_score", "probability"),
    )
    outcome_col = _first_present_column(predictions, ("y_true", "actual", "target", "realized_return", "forward_return"))
    if date_col is None or asset_col is None or score_col is None or outcome_col is None:
        return None

    frame = (
        predictions.select([date_col, asset_col, score_col, outcome_col])
        .rename({date_col: "date", asset_col: "asset", score_col: "score", outcome_col: "outcome"})
        .with_columns(pl.col("date").cast(pl.Date, strict=False))
        .filter(pl.col("date").is_not_null() & pl.col("score").is_not_null() & pl.col("outcome").is_not_null())
    )
    if frame.is_empty():
        return None

    daily_ic = _compute_daily_ic(frame)
    if daily_ic.is_empty():
        return None

    theme_config = get_theme_config(theme)
    ic_color = theme_config["colorway"][0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_ic["date"].to_list(), y=daily_ic["ic"].to_list(),
        mode="lines", name="Daily IC",
        line={"width": 0.8, "color": "rgba(10,22,40,0.25)"},
        showlegend=False,
    ))
    if "rolling_ic_21" in daily_ic.columns:
        fig.add_trace(go.Scatter(
            x=daily_ic["date"].to_list(), y=daily_ic["rolling_ic_21"].to_list(),
            mode="lines", name="21d Rolling IC",
            line={"width": 2, "color": ic_color},
            showlegend=False,
        ))
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"], line_width=0.5)
    fig.update_layout(theme_config["layout"])
    fig.update_layout(
        title="Daily Information Coefficient",
        height=300,
        margin={"t": 40, "l": 48, "r": 24, "b": 32},
        yaxis_title="IC",
    )
    return fig


def plot_quintile_returns(
    profile: BacktestProfile,
    *,
    theme: Literal["default", "dark", "print", "presentation"] = "default",
    n_quantiles: int = 10,
) -> go.Figure | None:
    """Plot realized return by prediction quantile — compact bar chart."""
    validate_theme(theme)
    predictions = profile.predictions_df
    if predictions.is_empty():
        return None

    date_col = _first_present_column(predictions, ("timestamp", "date", "session_date"))
    asset_col = _first_present_column(predictions, ("asset",))
    score_col = _first_present_column(
        predictions,
        ("prediction_value", "score", "prediction", "y_pred", "y_score", "ml_score", "probability"),
    )
    outcome_col = _first_present_column(predictions, ("y_true", "actual", "target", "realized_return", "forward_return"))
    if date_col is None or asset_col is None or score_col is None or outcome_col is None:
        return None

    frame = (
        predictions.select([date_col, asset_col, score_col, outcome_col])
        .rename({date_col: "date", asset_col: "asset", score_col: "score", outcome_col: "outcome"})
        .with_columns(pl.col("date").cast(pl.Date, strict=False))
        .filter(pl.col("date").is_not_null() & pl.col("score").is_not_null() & pl.col("outcome").is_not_null())
    )
    if frame.is_empty():
        return None

    quantile_summary = _compute_quantile_summary(frame, n_quantiles=n_quantiles)
    if quantile_summary.is_empty():
        return None

    q_labels = quantile_summary["quantile_label"].to_list()
    q_means = quantile_summary["mean_outcome"].to_list()
    q_counts = (
        quantile_summary["count"].to_list()
        if "count" in quantile_summary.columns
        else [None] * len(q_labels)
    )
    bar_colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in q_means]
    text_labels = [f"N={c:,}" if c is not None else "" for c in q_counts]

    theme_config = get_theme_config(theme)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=q_labels, y=q_means, marker_color=bar_colors,
        text=text_labels, textposition="outside", textfont={"size": 9},
        showlegend=False,
    ))
    fig.update_layout(theme_config["layout"])
    fig.update_layout(
        title="Mean Realized Return by Prediction Decile",
        height=300,
        margin={"t": 40, "l": 48, "r": 24, "b": 48},
        xaxis={"title": "Prediction Score Decile (D1=lowest, D10=highest)"},
        yaxis={"title": "Mean Forward Return", "tickformat": ".2%"},
    )
    return fig


def plot_prediction_trade_alignment(
    profile: BacktestProfile,
    *,
    theme: Literal["default", "dark", "print", "presentation"] = "default",
) -> go.Figure | None:
    """Plot trade outcome by entry-time prediction decile.

    Shows mean trade return per prediction decile at entry, with trade count
    annotations — directly answers whether stronger predictions lead to better
    trades.
    """
    validate_theme(theme)
    trades_df = profile.prediction_enriched_trades_df
    if trades_df.is_empty():
        return None

    # Prefer return % over $ PnL for comparability
    outcome_col = "pnl"
    outcome_label = "Mean Trade Return"
    outcome_fmt = ".2%"
    for candidate in ("pnl_pct", "return_pct", "return"):
        if candidate in trades_df.columns:
            outcome_col = candidate
            break
    if outcome_col == "pnl":
        outcome_label = "Mean Trade PnL"
        outcome_fmt = "$,.0f"
    if outcome_col not in trades_df.columns:
        return None

    entry_columns = [
        column
        for column in trades_df.columns
        if column.startswith("entry_") and trades_df.schema[column].is_numeric()
    ]
    if not entry_columns:
        return None

    value_col = _choose_primary_entry_column(entry_columns)
    aligned = trades_df.filter(
        pl.col(value_col).is_not_null() & pl.col(outcome_col).is_not_null()
    )
    if aligned.height < 5:
        return None

    # Bin trades by entry prediction into deciles
    n_bins = min(10, max(3, aligned.height // 5))
    scores = aligned[value_col].to_numpy()
    outcomes = aligned[outcome_col].to_numpy()
    edges = np.percentile(scores, np.linspace(0, 100, n_bins + 1))

    labels, means, counts, win_rates = [], [], [], []
    for j in range(n_bins):
        lo, hi = edges[j], edges[j + 1]
        mask = (scores >= lo) & (scores <= hi) if j == n_bins - 1 else (scores >= lo) & (scores < hi)
        if mask.sum() == 0:
            continue
        prefix = "D" if n_bins > 5 else "Q"
        labels.append(f"{prefix}{j + 1}")
        means.append(float(np.mean(outcomes[mask])))
        counts.append(int(mask.sum()))
        win_rates.append(float(np.mean(outcomes[mask] >= 0)))

    if not labels:
        return None

    bar_colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in means]
    text_labels = [f"N={c:,}<br>Win {wr:.0%}" for c, wr in zip(counts, win_rates)]

    theme_config = get_theme_config(theme)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=means, marker_color=bar_colors,
        text=text_labels, textposition="outside", textfont={"size": 9},
        showlegend=False,
    ))
    fig.update_layout(theme_config["layout"])
    pred_label = value_col.replace("entry_", "").replace("_", " ").title()
    fig.update_layout(
        title=f"Trade Outcome by Entry {pred_label} Decile",
        height=300,
        margin={"t": 40, "l": 48, "r": 24, "b": 48},
        xaxis={"title": f"Entry {pred_label} Decile (D1=lowest, D{n_bins}=highest)"},
        yaxis={"title": outcome_label, "tickformat": outcome_fmt},
    )
    return fig


def _choose_primary_entry_column(columns: list[str]) -> str:
    preferred = (
        "entry_prediction_value",
        "entry_prediction",
        "entry_y_score",
        "entry_ml_score",
        "entry_score",
        "entry_y_pred",
        "entry_signal",
        "entry_target_weight",
    )
    for candidate in preferred:
        if candidate in columns:
            return candidate
    return columns[0]


def _first_present_column(df: pl.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _compute_daily_ic(frame: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for date_value, group in frame.group_by("date", maintain_order=True):
        if group.height < 3:
            continue
        score = group["score"].to_numpy()
        outcome = group["outcome"].to_numpy()
        ic = _spearman_correlation(score, outcome)
        if ic is None or np.isnan(ic):
            continue
        normalized_date = date_value[0] if isinstance(date_value, tuple) else date_value
        rows.append({"date": normalized_date, "ic": float(ic)})
    if not rows:
        return pl.DataFrame(schema={"date": pl.Date, "ic": pl.Float64(), "rolling_ic_21": pl.Float64()})
    return (
        pl.DataFrame(rows)
        .sort("date")
        .with_columns(pl.col("ic").rolling_mean(window_size=21, min_samples=5).alias("rolling_ic_21"))
    )


def _compute_quantile_summary(frame: pl.DataFrame, *, n_quantiles: int) -> pl.DataFrame:
    ranked = (
        frame.sort(["date", "score"])
        .with_columns(
            (
                (
                    (pl.col("score").rank("ordinal").over("date") - 1)
                    / pl.len().over("date").clip(lower_bound=1)
                    * n_quantiles
                )
                .floor()
                .clip(0, n_quantiles - 1)
                .cast(pl.Int32)
                + 1
            ).alias("quantile")
        )
    )
    summary = (
        ranked.group_by("quantile")
        .agg(
            pl.col("outcome").mean().alias("mean_outcome"),
            pl.col("outcome").len().alias("count"),
        )
        .sort("quantile")
        .with_columns(
            pl.when(n_quantiles <= 5)
            .then(pl.format("Q{}", pl.col("quantile")))
            .otherwise(pl.format("D{}", pl.col("quantile")))
            .alias("quantile_label")
        )
    )
    return summary


def _compute_regime_summary(profile: BacktestProfile, daily_ic: pl.DataFrame) -> pl.DataFrame:
    equity = profile.equity_df
    if equity.is_empty() or "timestamp" not in equity.columns:
        return pl.DataFrame()
    # Compute daily return if missing
    if "return" in equity.columns:
        ret_col = equity["return"]
    elif "cumulative_return" in equity.columns:
        cum = equity["cumulative_return"]
        ret_col = (1 + cum) / (1 + cum.shift(1)) - 1
    elif "equity" in equity.columns:
        eq_val = equity["equity"]
        ret_col = eq_val / eq_val.shift(1) - 1
    else:
        return pl.DataFrame()
    regime_frame = (
        equity.select(
            pl.col("timestamp").cast(pl.Date, strict=False).alias("date"),
        ).with_columns(ret_col.alias("strategy_return"))
        .with_columns(
            pl.col("strategy_return")
            .rolling_std(window_size=63, min_samples=20)
            .alias("rolling_vol_63")
        )
    )
    median_vol = (
        float(regime_frame["rolling_vol_63"].drop_nulls().median())
        if regime_frame["rolling_vol_63"].drop_nulls().len() > 0
        else 0.0
    )
    regime_frame = regime_frame.with_columns(
        [
            pl.when(pl.col("rolling_vol_63").is_not_null() & (pl.col("rolling_vol_63") >= median_vol))
            .then(pl.lit("High Vol"))
            .otherwise(pl.lit("Low Vol"))
            .alias("vol_regime"),
            pl.when(pl.col("strategy_return") >= 0)
            .then(pl.lit("Up Days"))
            .otherwise(pl.lit("Down Days"))
            .alias("return_regime"),
        ]
    )
    joined = daily_ic.join(regime_frame, on="date", how="left")
    rows: list[dict[str, object]] = []
    for column in ("vol_regime", "return_regime"):
        if column not in joined.columns:
            continue
        summary = joined.group_by(column).agg(pl.col("ic").mean().alias("mean_ic")).sort(column)
        for row in summary.iter_rows(named=True):
            rows.append({"regime": row[column], "mean_ic": row["mean_ic"]})
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _spearman_correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) != len(y) or len(x) < 3:
        return None
    x_rank = pl.Series(x).rank("average").to_numpy()
    y_rank = pl.Series(y).rank("average").to_numpy()
    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return None
    return float(np.corrcoef(x_rank, y_rank)[0, 1])
