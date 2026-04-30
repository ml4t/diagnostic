"""Uncertainty quantification for cross-sectional IC and AUC time series.

The estimands here treat the **date** as the unit of observation, not the
individual asset prediction. For a daily cross-sectional ranking strategy this
matches the metric of interest ("how well does the model rank assets on a
typical day?") and avoids mixing time-series and cross-sectional variation.

Three uncertainty estimates are reported side by side:

1. Naive standard error of the mean (assumes independent daily metrics).
2. HAC / Newey-West standard error (autocorrelation-robust; preferred default).
3. Stationary block bootstrap percentile interval (nonparametric robustness
   check).

The HAC lag should match or exceed the forward-return horizon, since
overlapping forward returns mechanically induce serial dependence in the
daily-metric series. Callers pass ``horizon`` and the helper picks
``max(horizon - 1, NW_auto)``.

For per-date AUC across a cross-section, ``cross_sectional_auc_series`` uses
the rank-based Mann-Whitney U formula in pure Polars, with no Python loop
over dates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

from ml4t.diagnostic.metrics.ic_inference import compute_ic_hac_stats

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Cross-sectional AUC series
# ---------------------------------------------------------------------------


def cross_sectional_auc_series(
    predictions: pl.DataFrame | pd.DataFrame,
    labels: pl.DataFrame | pd.DataFrame,
    pred_col: str = "prediction",
    label_col: str = "label",
    date_col: str = "date",
    entity_col: str | list[str] | None = None,
    min_obs: int = 10,
) -> pl.DataFrame | pd.DataFrame:
    """Compute a per-date cross-sectional AUC time series for binary labels.

    AUC is computed via the Mann-Whitney U identity, which is equivalent to
    ranking all scores within the date and using

    .. math::

        \\mathrm{AUC} = \\frac{R_+ - n_+(n_+ + 1)/2}{n_+ \\, n_-}

    where :math:`R_+` is the sum of ranks of positive examples and
    :math:`n_+, n_-` are the positive/negative counts. This is fully
    vectorized in Polars (rank ``over(date)`` then group-by aggregate), so it
    avoids ``sklearn.metrics.roc_auc_score`` per-date Python loops.

    Parameters
    ----------
    predictions, labels
        DataFrames with ``date_col`` (and ``entity_col`` if panel data),
        ``pred_col`` continuous score, ``label_col`` binary ``{0, 1}`` label.
        May be the same DataFrame.
    min_obs
        Minimum cross-section size for a valid per-date AUC. Dates with fewer
        rows or with no positive or no negative examples emit ``ic = null``.

    Returns
    -------
    DataFrame with columns ``[date_col, "auc", "n_obs", "n_pos"]``. Type
    matches the input (polars in → polars out, pandas in → pandas out).
    """
    output_as_pandas = isinstance(predictions, pd.DataFrame)

    join_on: list[str] = [date_col]
    if entity_col is not None:
        if isinstance(entity_col, str):
            join_on.append(entity_col)
        else:
            join_on.extend(entity_col)

    pred_pl = (
        predictions
        if isinstance(predictions, pl.DataFrame)
        else pl.from_pandas(cast(pd.DataFrame, predictions))
    )
    label_pl = (
        labels if isinstance(labels, pl.DataFrame) else pl.from_pandas(cast(pd.DataFrame, labels))
    )

    if pred_col in label_pl.columns and pred_col not in pred_pl.columns:
        # Caller passed a single combined DataFrame as both arguments.
        pred_pl = label_pl

    if label_col in pred_pl.columns and pred_pl is label_pl:
        df = pred_pl
    else:
        # Avoid duplicating columns when both inputs are the same DataFrame.
        if pred_pl is label_pl:
            df = pred_pl
        else:
            df = pred_pl.join(label_pl, on=join_on, how="inner")

    valid_expr = pl.col(pred_col).is_finite() & pl.col(label_col).is_not_null()

    df_valid = df.with_columns(
        [
            pl.when(valid_expr).then(pl.col(pred_col)).otherwise(None).alias("__score"),
            pl.when(valid_expr)
            .then(pl.col(label_col).cast(pl.Float64))
            .otherwise(None)
            .alias("__y"),
        ]
    ).with_columns(
        pl.col("__score").rank(method="average").over(date_col).alias("__rank"),
    )

    all_dates = df.select(date_col).unique().sort(date_col)

    grouped = df_valid.group_by(date_col, maintain_order=False).agg(
        [
            valid_expr.sum().alias("n_obs"),
            pl.col("__y").sum().alias("n_pos"),
            (pl.col("__rank") * pl.col("__y")).sum().alias("__r_pos"),
        ]
    )

    auc_pl = (
        all_dates.join(grouped, on=date_col, how="left")
        .with_columns(
            [
                pl.col("n_obs").fill_null(0),
                pl.col("n_pos").fill_null(0),
                (pl.col("n_obs") - pl.col("n_pos")).alias("n_neg"),
            ]
        )
        .with_columns(
            pl.when((pl.col("n_obs") >= min_obs) & (pl.col("n_pos") > 0) & (pl.col("n_neg") > 0))
            .then(
                (pl.col("__r_pos") - pl.col("n_pos") * (pl.col("n_pos") + 1) / 2)
                / (pl.col("n_pos") * pl.col("n_neg"))
            )
            .otherwise(None)
            .alias("auc")
        )
        .select([date_col, "auc", "n_obs", "n_pos"])
        .sort(date_col)
    )

    if output_as_pandas:
        return auc_pl.to_pandas()
    return auc_pl


# ---------------------------------------------------------------------------
# Block bootstrap of the mean of a daily metric series
# ---------------------------------------------------------------------------


def _stationary_bootstrap_block_indices(
    n: int, block_size: float, rng: np.random.Generator
) -> NDArray[np.int_]:
    """Generate one stationary-bootstrap index sequence of length ``n``."""
    p = 1.0 / max(block_size, 1.0)
    indices = np.empty(n, dtype=np.int_)
    i = 0
    while i < n:
        start = int(rng.integers(0, n))
        block_len = int(rng.geometric(p))
        end = min(i + block_len, n)
        for k in range(end - i):
            indices[i + k] = (start + k) % n
        i = end
    return indices


def _block_bootstrap_mean_ci(
    values: NDArray[Any],
    *,
    block_size: float,
    n_boot: int,
    alpha: float,
    seed: int = 0,
) -> dict[str, float]:
    """Block-bootstrap percentile CI for the mean of a univariate series."""
    if values.size < 3:
        return {
            "mean": float(np.nan),
            "ci_lower": float(np.nan),
            "ci_upper": float(np.nan),
            "block_size": float(block_size),
            "n_boot": 0,
        }

    rng = np.random.default_rng(seed)
    n = values.size
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = _stationary_bootstrap_block_indices(n, block_size, rng)
        means[b] = float(np.mean(values[idx]))

    lo = float(np.percentile(means, 100 * (alpha / 2)))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return {
        "mean": float(np.mean(values)),
        "ci_lower": lo,
        "ci_upper": hi,
        "block_size": float(block_size),
        "n_boot": int(n_boot),
    }


# ---------------------------------------------------------------------------
# Daily-metric uncertainty wrapper (IC and AUC)
# ---------------------------------------------------------------------------


def _to_daily_array(
    daily_series: pl.DataFrame | pd.DataFrame | pl.Series | pd.Series | NDArray[Any],
    value_col: str,
) -> NDArray[Any]:
    """Extract the daily metric values as a 1-D float array."""
    if isinstance(daily_series, pl.DataFrame):
        arr = daily_series[value_col].to_numpy()
    elif isinstance(daily_series, pd.DataFrame):
        arr = daily_series[value_col].to_numpy()
    elif isinstance(daily_series, pl.Series | pd.Series):
        arr = np.asarray(daily_series)
    else:
        arr = np.asarray(daily_series).flatten()
    arr = np.asarray(arr, dtype=np.float64).flatten()
    return arr[np.isfinite(arr)]


def _newey_west_lag(n: int, horizon: int) -> int:
    """HAC lag = max(horizon - 1, Newey-West auto), capped at n // 2."""
    nw_auto = int(np.floor(4 * (max(n, 1) / 100) ** (2 / 9)))
    nw_auto = max(1, nw_auto)
    base = max(horizon - 1, nw_auto)
    return max(1, min(base, max(1, n // 2)))


def compute_ic_uncertainty(
    daily_ic: pl.DataFrame | pd.DataFrame | pl.Series | pd.Series | NDArray[Any],
    horizon: int = 1,
    *,
    ic_col: str = "ic",
    alpha: float = 0.05,
    n_boot: int = 2000,
    block_size: float | None = None,
    kernel: str = "bartlett",
    seed: int = 0,
) -> dict[str, float | int]:
    """Bundle naive, HAC, and block-bootstrap uncertainty for a daily-IC series.

    The input is the per-date IC series produced by
    :func:`cross_sectional_ic_series`, optionally pooled across folds. Each
    valid (non-null) row is treated as one observation of the daily IC.

    The HAC lag is ``max(horizon - 1, Newey-West auto)`` because overlapping
    forward returns of horizon ``H`` induce up to ``H - 1`` lags of serial
    dependence in the IC. The block bootstrap uses an expected block size of
    at least ``horizon``.

    Returns
    -------
    dict with keys
        ``mean_ic, std_ic, n_days, pct_positive``,
        ``se_naive, ci_naive_lower, ci_naive_upper``,
        ``se_hac, ci_hac_lower, ci_hac_upper, t_hac, p_hac, hac_lag``,
        ``ci_boot_lower, ci_boot_upper, boot_block_size, n_boot``.
    """
    values = _to_daily_array(daily_ic, ic_col)
    n = int(values.size)

    out: dict[str, float | int] = {
        "mean_ic": float(np.nan),
        "std_ic": float(np.nan),
        "n_days": n,
        "pct_positive": float(np.nan),
        "se_naive": float(np.nan),
        "ci_naive_lower": float(np.nan),
        "ci_naive_upper": float(np.nan),
        "se_hac": float(np.nan),
        "ci_hac_lower": float(np.nan),
        "ci_hac_upper": float(np.nan),
        "t_hac": float(np.nan),
        "p_hac": float(np.nan),
        "hac_lag": 0,
        "ci_boot_lower": float(np.nan),
        "ci_boot_upper": float(np.nan),
        "boot_block_size": float("nan"),
        "n_boot": 0,
    }

    if n < 3:
        return out

    mean_ic = float(np.mean(values))
    std_ic = float(np.std(values, ddof=1))
    out["mean_ic"] = mean_ic
    out["std_ic"] = std_ic
    out["pct_positive"] = float(np.mean(values > 0))

    # Naive SE / CI (assumes independence — included as a sanity baseline).
    se_naive = std_ic / np.sqrt(n) if std_ic > 0 else float(np.nan)
    out["se_naive"] = float(se_naive)
    if np.isfinite(se_naive) and se_naive > 0:
        t_crit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
        out["ci_naive_lower"] = mean_ic - t_crit * se_naive
        out["ci_naive_upper"] = mean_ic + t_crit * se_naive

    # HAC SE / CI via existing compute_ic_hac_stats.
    hac_lag = _newey_west_lag(n, int(max(1, horizon)))
    hac = compute_ic_hac_stats(
        pl.DataFrame({ic_col: values}),
        ic_col=ic_col,
        maxlags=hac_lag,
        kernel=kernel,
    )
    se_hac = float(hac["hac_se"])
    out["se_hac"] = se_hac
    out["t_hac"] = float(hac["t_stat"])
    out["p_hac"] = float(hac["p_value"])
    out["hac_lag"] = int(hac["effective_lags"])
    if np.isfinite(se_hac) and se_hac > 0:
        t_crit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
        out["ci_hac_lower"] = mean_ic - t_crit * se_hac
        out["ci_hac_upper"] = mean_ic + t_crit * se_hac

    # Block bootstrap. Block size defaults to the larger of the horizon and
    # n^{1/3}, which is a standard rule of thumb for stationary bootstrap.
    if block_size is None:
        block_size = float(max(int(max(1, horizon)), max(1, int(round(n ** (1 / 3))))))
    boot = _block_bootstrap_mean_ci(
        values,
        block_size=float(block_size),
        n_boot=int(n_boot),
        alpha=alpha,
        seed=seed,
    )
    out["ci_boot_lower"] = boot["ci_lower"]
    out["ci_boot_upper"] = boot["ci_upper"]
    out["boot_block_size"] = boot["block_size"]
    out["n_boot"] = boot["n_boot"]

    return out


def compute_auc_uncertainty(
    daily_auc: pl.DataFrame | pd.DataFrame | pl.Series | pd.Series | NDArray[Any],
    horizon: int = 1,
    *,
    auc_col: str = "auc",
    null_value: float = 0.5,
    alpha: float = 0.05,
    n_boot: int = 2000,
    block_size: float | None = None,
    kernel: str = "bartlett",
    seed: int = 0,
) -> dict[str, float | int]:
    """Daily-AUC analogue of :func:`compute_ic_uncertainty`.

    AUC is centered on ``null_value`` (default 0.5) for the HAC t-statistic
    and p-value, since the relevant null hypothesis is "AUC = chance" rather
    than "AUC = 0". All other quantities mirror the IC version.

    Returns the same keys as :func:`compute_ic_uncertainty` with ``ic`` →
    ``auc`` (e.g. ``mean_auc``, ``ci_hac_lower``, ``ci_hac_upper``, ...).
    """
    values = _to_daily_array(daily_auc, auc_col)
    n = int(values.size)

    out: dict[str, float | int] = {
        "mean_auc": float(np.nan),
        "std_auc": float(np.nan),
        "n_days": n,
        "pct_above_null": float(np.nan),
        "se_naive": float(np.nan),
        "ci_naive_lower": float(np.nan),
        "ci_naive_upper": float(np.nan),
        "se_hac": float(np.nan),
        "ci_hac_lower": float(np.nan),
        "ci_hac_upper": float(np.nan),
        "t_hac": float(np.nan),
        "p_hac": float(np.nan),
        "hac_lag": 0,
        "ci_boot_lower": float(np.nan),
        "ci_boot_upper": float(np.nan),
        "boot_block_size": float("nan"),
        "n_boot": 0,
    }

    if n < 3:
        return out

    mean_auc = float(np.mean(values))
    std_auc = float(np.std(values, ddof=1))
    out["mean_auc"] = mean_auc
    out["std_auc"] = std_auc
    out["pct_above_null"] = float(np.mean(values > null_value))

    se_naive = std_auc / np.sqrt(n) if std_auc > 0 else float(np.nan)
    out["se_naive"] = float(se_naive)
    if np.isfinite(se_naive) and se_naive > 0:
        t_crit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
        out["ci_naive_lower"] = mean_auc - t_crit * se_naive
        out["ci_naive_upper"] = mean_auc + t_crit * se_naive

    # HAC: feed (auc - null_value) so the t-statistic tests AUC = null_value.
    hac_lag = _newey_west_lag(n, int(max(1, horizon)))
    centered = pl.DataFrame({auc_col: values - null_value})
    hac = compute_ic_hac_stats(
        centered,
        ic_col=auc_col,
        maxlags=hac_lag,
        kernel=kernel,
    )
    se_hac = float(hac["hac_se"])
    out["se_hac"] = se_hac
    out["t_hac"] = float(hac["t_stat"])
    out["p_hac"] = float(hac["p_value"])
    out["hac_lag"] = int(hac["effective_lags"])
    if np.isfinite(se_hac) and se_hac > 0:
        t_crit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
        out["ci_hac_lower"] = mean_auc - t_crit * se_hac
        out["ci_hac_upper"] = mean_auc + t_crit * se_hac

    if block_size is None:
        block_size = float(max(int(max(1, horizon)), max(1, int(round(n ** (1 / 3))))))
    boot = _block_bootstrap_mean_ci(
        values,
        block_size=float(block_size),
        n_boot=int(n_boot),
        alpha=alpha,
        seed=seed,
    )
    out["ci_boot_lower"] = boot["ci_lower"]
    out["ci_boot_upper"] = boot["ci_upper"]
    out["boot_block_size"] = boot["block_size"]
    out["n_boot"] = boot["n_boot"]

    return out


__all__ = [
    "cross_sectional_auc_series",
    "compute_ic_uncertainty",
    "compute_auc_uncertainty",
]
