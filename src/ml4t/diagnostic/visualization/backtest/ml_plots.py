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
    """Plot how entry-time prediction values align with realized trade outcomes."""
    validate_theme(theme)
    trades_df = profile.prediction_enriched_trades_df
    if trades_df.is_empty():
        return None

    # Prefer return % over $ PnL for comparability with prediction scores
    outcome_col = "pnl"
    outcome_label = "Trade PnL"
    outcome_fmt = "$,.0f"
    for candidate in ("pnl_pct", "return_pct", "return"):
        if candidate in trades_df.columns:
            outcome_col = candidate
            outcome_label = "Trade Return"
            outcome_fmt = ".1%"
            break
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
    aligned = (
        trades_df
        .filter(pl.col(value_col).is_not_null() & pl.col(outcome_col).is_not_null())
        .with_columns(
            pl.when(pl.col(outcome_col) >= 0)
            .then(pl.lit("Winning trades"))
            .otherwise(pl.lit("Losing trades"))
            .alias("_outcome_bucket")
        )
    )
    if aligned.is_empty():
        return None

    theme_config = get_theme_config(theme)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Entry Prediction Distribution", f"Entry Prediction vs {outcome_label}"),
        horizontal_spacing=0.12,
    )

    colors = {
        "Winning trades": COLORS["positive"],
        "Losing trades": COLORS["negative"],
    }
    for bucket in ("Winning trades", "Losing trades"):
        subset = aligned.filter(pl.col("_outcome_bucket") == bucket)
        if subset.is_empty():
            continue
        fig.add_trace(
            go.Histogram(
                x=subset[value_col].to_list(),
                name=bucket,
                marker_color=colors[bucket],
                opacity=0.7,
                nbinsx=min(30, max(10, int(np.sqrt(subset.height)))),
                showlegend=True,
                legendgroup=bucket,
            ),
            row=1,
            col=1,
        )
        hover_fmt = "%{y:.2%}" if outcome_fmt == ".1%" else "%{y:$,.0f}"
        fig.add_trace(
            go.Scatter(
                x=subset[value_col].to_list(),
                y=subset[outcome_col].to_list(),
                mode="markers",
                name=bucket,
                marker={
                    "color": colors[bucket],
                    "size": 8,
                    "opacity": 0.75,
                },
                legendgroup=bucket,
                showlegend=False,
                hovertemplate=(
                    f"{value_col}: %{{x:.4f}}<br>"
                    f"{outcome_label}: {hover_fmt}<br>"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

    # Add binned-mean overlay on scatter
    score_arr = aligned[value_col].to_numpy()
    outcome_arr = aligned[outcome_col].to_numpy()
    if len(score_arr) >= 10:
        n_bins = min(10, max(5, len(score_arr) // 10))
        edges = np.percentile(score_arr, np.linspace(0, 100, n_bins + 1))
        bin_centers: list[float] = []
        bin_means: list[float] = []
        for j in range(n_bins):
            mask = (score_arr >= edges[j]) & (score_arr <= edges[j + 1])
            if mask.sum() > 0:
                bin_centers.append(float(np.mean(score_arr[mask])))
                bin_means.append(float(np.mean(outcome_arr[mask])))
        if bin_centers:
            fig.add_trace(
                go.Scatter(
                    x=bin_centers, y=bin_means,
                    mode="lines+markers",
                    name="Binned Mean",
                    line={"color": COLORS["amber"], "width": 2.5},
                    marker={"size": 8, "color": COLORS["amber"]},
                    showlegend=True,
                ),
                row=1, col=2,
            )

    fig.update_layout(
        title="Prediction-to-Trade Alignment",
        barmode="overlay",
        height=300,
        template=theme_config.get("template", "plotly_white"),
        paper_bgcolor=theme_config.get("paper_bgcolor"),
        plot_bgcolor=theme_config.get("plot_bgcolor"),
        font={"color": theme_config.get("font_color")},
        margin={"l": 40, "r": 24, "t": 72, "b": 48},
    )
    fig.update_xaxes(title_text=value_col.replace("entry_", "").replace("_", " ").title(), row=1, col=1)
    fig.update_xaxes(title_text=value_col.replace("entry_", "").replace("_", " ").title(), row=1, col=2)
    fig.update_yaxes(title_text="Trades", row=1, col=1)
    fig.update_yaxes(
        title_text=outcome_label, row=1, col=2,
        tickformat=outcome_fmt if outcome_fmt == ".1%" else None,
    )
    return fig


def plot_prediction_calibration(
    profile: BacktestProfile,
    *,
    n_bins: int = 10,
    theme: Literal["default", "dark", "print", "presentation"] = "default",
) -> go.Figure | None:
    """Plot model calibration — predicted score vs actual positive rate.

    Bins prediction scores into quantiles and computes the actual win rate
    (PnL > 0) per bin.  Perfect calibration lies on the diagonal.
    """
    validate_theme(theme)
    predictions = profile.predictions_df
    if predictions.is_empty():
        return None

    score_col = _first_present_column(
        predictions,
        ("prediction_value", "score", "prediction", "y_pred", "y_score", "probability"),
    )
    outcome_col = _first_present_column(
        predictions, ("y_true", "actual", "target", "realized_return", "forward_return"),
    )
    if score_col is None or outcome_col is None:
        return None

    frame = (
        predictions.select([score_col, outcome_col])
        .rename({score_col: "score", outcome_col: "outcome"})
        .filter(pl.col("score").is_not_null() & pl.col("outcome").is_not_null())
    )
    if frame.height < n_bins * 3:
        return None

    # Bin by score quantile
    binned = frame.with_columns(
        (
            (
                (pl.col("score").rank("ordinal") - 1)
                / pl.len().clip(lower_bound=1)
                * n_bins
            )
            .floor()
            .clip(0, n_bins - 1)
            .cast(pl.Int32)
            + 1
        ).alias("bin")
    )

    summary = (
        binned.group_by("bin")
        .agg([
            pl.col("score").mean().alias("mean_score"),
            (pl.col("outcome") > 0).mean().alias("actual_positive_rate"),
            pl.len().alias("count"),
        ])
        .sort("bin")
    )

    theme_config = get_theme_config(theme)
    fig = go.Figure()

    # Build clean bin labels (D1..D10) instead of raw score values
    bin_labels = [f"D{b}" for b in summary["bin"].to_list()]
    win_rates = summary["actual_positive_rate"].to_list()
    counts = summary["count"].to_list()
    mean_scores = summary["mean_score"].to_list()

    hover_text = [
        f"Bin {label}<br>Win Rate: {wr:.1%}<br>Avg Score: {ms:.4f}<br>N: {n:,}"
        for label, wr, ms, n in zip(bin_labels, win_rates, mean_scores, counts)
    ]

    fig.add_trace(go.Bar(
        x=bin_labels,
        y=win_rates,
        name="Actual Win Rate",
        marker_color=theme_config["colorway"][0],
        opacity=0.8,
        text=[f"{wr:.0%}" for wr in win_rates],
        textposition="outside",
        textfont={"size": 9},
        hovertext=hover_text,
        hoverinfo="text",
    ))

    # Random baseline at 50%
    fig.add_hline(
        y=0.5, line_dash="dash", line_color="gray", line_width=1.5,
        annotation_text="Random (50%)", annotation_position="right",
        annotation_font_size=10, annotation_font_color="gray",
    )

    fig.update_layout(
        title="Prediction Calibration",
        xaxis_title="Prediction Score Decile",
        yaxis_title="Actual Win Rate",
        yaxis_tickformat=".0%",
        height=300,
        template=theme_config.get("template", "plotly_white"),
        paper_bgcolor=theme_config.get("paper_bgcolor"),
        plot_bgcolor=theme_config.get("plot_bgcolor"),
        font={"color": theme_config.get("font_color")},
        margin={"l": 40, "r": 24, "t": 72, "b": 48},
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
