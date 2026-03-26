"""ML-oriented backtest plots built from prediction surfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

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
    outcome_col = _first_present_column(predictions, ("y_true", "target", "realized_return", "forward_return"))
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
    quantile_summary = _compute_quantile_summary(frame, n_quantiles=5)
    regime_summary = _compute_regime_summary(profile, daily_ic)

    theme_config = get_theme_config(theme)
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "Daily Information Coefficient",
            "Realized Return by Prediction Quintile",
            "Information Coefficient by Strategy Regime",
        ),
        vertical_spacing=0.1,
        row_heights=[0.46, 0.24, 0.3],
    )

    fig.add_trace(
        go.Scatter(
            x=daily_ic["date"].to_list(),
            y=daily_ic["ic"].to_list(),
            mode="lines",
            name="Daily IC",
            line={"width": 1.5},
        ),
        row=1,
        col=1,
    )
    if "rolling_ic_21" in daily_ic.columns:
        fig.add_trace(
            go.Scatter(
                x=daily_ic["date"].to_list(),
                y=daily_ic["rolling_ic_21"].to_list(),
                mode="lines",
                name="21d Mean IC",
                line={"width": 2.5},
            ),
            row=1,
            col=1,
        )
    if not quantile_summary.is_empty():
        fig.add_trace(
            go.Bar(
                x=quantile_summary["quantile_label"].to_list(),
                y=quantile_summary["mean_outcome"].to_list(),
                name="Mean Realized Return",
                marker_color=theme_config["colorway"][0],
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    if not regime_summary.is_empty():
        fig.add_trace(
            go.Bar(
                x=regime_summary["regime"].to_list(),
                y=regime_summary["mean_ic"].to_list(),
                name="Mean IC",
                marker_color=theme_config["colorway"][1],
                showlegend=False,
            ),
            row=3,
            col=1,
        )

    fig.update_layout(
        title="Prediction Diagnostics",
        height=980,
        template=theme_config.get("template", "plotly_white"),
        paper_bgcolor=theme_config.get("paper_bgcolor"),
        plot_bgcolor=theme_config.get("plot_bgcolor"),
        font={"color": theme_config.get("font_color")},
        margin={"l": 40, "r": 24, "t": 72, "b": 48},
    )
    fig.update_yaxes(title_text="IC", row=1, col=1)
    fig.update_yaxes(title_text="Mean Outcome", tickformat=".2%", row=2, col=1)
    fig.update_yaxes(title_text="Mean IC", row=3, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, row=3, col=1)
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
        "Winning trades": "#2f855a",
        "Losing trades": "#c53030",
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
                    line={"color": "#2b6cb0", "width": 2.5},
                    marker={"size": 8, "color": "#2b6cb0"},
                    showlegend=True,
                ),
                row=1, col=2,
            )

    fig.update_layout(
        title="Prediction-to-Trade Alignment",
        barmode="overlay",
        height=420,
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
        predictions, ("y_true", "target", "realized_return", "forward_return"),
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

    fig.add_trace(go.Bar(
        x=summary["mean_score"].to_list(),
        y=summary["actual_positive_rate"].to_list(),
        name="Actual Win Rate",
        marker_color=theme_config["colorway"][0],
        opacity=0.8,
    ))

    # Diagonal reference (perfect calibration)
    x_range = [float(summary["mean_score"].min()), float(summary["mean_score"].max())]
    fig.add_trace(go.Scatter(
        x=x_range, y=[0.5, 0.5],
        mode="lines", name="Random (50%)",
        line={"color": "gray", "dash": "dash", "width": 1.5},
    ))

    fig.update_layout(
        title="Prediction Calibration",
        xaxis_title="Mean Prediction Score (by decile)",
        yaxis_title="Actual Win Rate",
        yaxis_tickformat=".0%",
        height=380,
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
        .agg(pl.col("outcome").mean().alias("mean_outcome"))
        .sort("quantile")
        .with_columns(pl.format("Q{}", pl.col("quantile")).alias("quantile_label"))
    )
    return summary


def _compute_regime_summary(profile: BacktestProfile, daily_ic: pl.DataFrame) -> pl.DataFrame:
    equity = profile.equity_df
    if equity.is_empty() or "timestamp" not in equity.columns or "return" not in equity.columns:
        return pl.DataFrame()
    regime_frame = (
        equity.select(
            [
                pl.col("timestamp").cast(pl.Date, strict=False).alias("date"),
                pl.col("return").alias("strategy_return"),
            ]
        )
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
