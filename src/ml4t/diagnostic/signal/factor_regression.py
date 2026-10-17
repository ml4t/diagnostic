"""Dated factor-portfolio regression against the same supplied asset universe."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from io import BytesIO
from numbers import Real

import numpy as np
import pandas as pd
import polars as pl

from ml4t.diagnostic.evaluation.portfolio_analysis.metrics import alpha_beta
from ml4t.diagnostic.signal._utils import _signal_frame, _signal_inputs
from ml4t.diagnostic.signal.portfolio import _weight_flags, compute_factor_returns
from ml4t.diagnostic.signal.quantile import _sample_return_statistics
from ml4t.diagnostic.utils.validation import (
    _align_datetime_key_resolution,
    _validate_observation_keys,
)


@dataclass(frozen=True)
class FactorAlphaBetaResult:
    """Owned regression statistics and dated samples, including undefined values.

    ``statistics`` contains one row per supplied return selector. ``observations``
    retains the actual portfolio date grid, both returns, finite asset counts,
    joint-sample mask, fitted values and residuals. Frames and metadata are fresh
    copies. JSON uses Arrow IPC snapshots to preserve timestamp units/timezones
    and nulls without nonstandard NaN/Infinity literals.
    """

    _statistics: bytes = field(repr=False)
    _observations: bytes = field(repr=False)
    _metadata: str = field(repr=False)

    @property
    def statistics(self) -> pl.DataFrame:
        """Return independent per-horizon regression statistics."""
        return pl.read_ipc(BytesIO(self._statistics))

    @property
    def observations(self) -> pl.DataFrame:
        """Return independent typed dates, samples and regression diagnostics."""
        return pl.read_ipc(BytesIO(self._observations))

    @property
    def metadata(self) -> dict:
        """Return independent workflow definitions and applied weight options."""
        return json.loads(self._metadata)

    def to_json(self) -> str:
        """Serialize owned results as strict JSON with typed Arrow tables."""
        return json.dumps(
            {
                "format_version": 1,
                "statistics_ipc_base64": base64.b64encode(self._statistics).decode("ascii"),
                "observations_ipc_base64": base64.b64encode(self._observations).decode("ascii"),
                "metadata": self.metadata,
            },
            allow_nan=False,
        )

    @classmethod
    def from_json(cls, value: str) -> FactorAlphaBetaResult:
        """Restore this versioned result, rejecting unsupported formats."""
        parsed = json.loads(value)
        if parsed.get("format_version") != 1:
            raise ValueError("unsupported factor alpha/beta result format_version")
        statistics = base64.b64decode(parsed["statistics_ipc_base64"], validate=True)
        observations = base64.b64decode(parsed["observations_ipc_base64"], validate=True)
        pl.read_ipc(BytesIO(statistics))
        pl.read_ipc(BytesIO(observations))
        return cls(statistics, observations, json.dumps(parsed["metadata"], allow_nan=False))


def _provided_returns(
    returns: pl.DataFrame | pd.DataFrame | pd.Series,
    columns: list[str],
    date_col: str,
) -> pl.DataFrame:
    if isinstance(returns, pd.Series):
        if len(columns) != 1:
            raise ValueError("a provided returns Series requires exactly one return column")
        returns = returns.rename(columns[0]).rename_axis(date_col).reset_index()
    elif isinstance(returns, pd.DataFrame) and date_col not in returns.columns:
        returns = returns.rename_axis(date_col).reset_index()
    values = _signal_frame(returns)
    _validate_observation_keys(values, [date_col], "provided factor returns")
    for column in columns:
        if column not in values.columns or not values[column].dtype.is_numeric():
            raise ValueError(f"provided factor returns require numeric column {column!r}")
    return values.select(pl.col(date_col).alias("date"), *columns)


def _universe_means(values: pl.DataFrame, column: str, date_col: str) -> pl.DataFrame:
    benchmark = (
        values.select(
            pl.col(date_col).alias("date"),
            pl.when(pl.col(column).cast(pl.Float64).is_finite())
            .then(pl.col(column).cast(pl.Float64))
            .otherwise(None)
            .alias("universe_return"),
        )
        .group_by("date")
        .agg(
            pl.col("universe_return").mean(),
            pl.col("universe_return").count().alias("n_universe"),
        )
    )
    overflow = benchmark.filter(
        (pl.col("n_universe") > 0) & ~pl.col("universe_return").is_finite().fill_null(False)
    )
    if overflow.is_empty():
        return benchmark
    corrections = []
    affected = values.join(overflow.select(pl.col("date").alias(date_col)), on=date_col, how="semi")
    for part in affected.partition_by(date_col, maintain_order=True):
        outcomes = part[column].cast(pl.Float64).to_numpy()
        mean = _sample_return_statistics(outcomes[np.isfinite(outcomes)])[0]
        corrections.append(
            part.head(1)
            .select(pl.col(date_col).alias("date"))
            .with_columns(pl.lit(mean).alias("__stable_mean"))
        )
    return (
        benchmark.join(pl.concat(corrections), on="date", how="left")
        .with_columns(pl.coalesce("__stable_mean", "universe_return").alias("universe_return"))
        .drop("__stable_mean")
    )


def _fit(part: pl.DataFrame, scale: float) -> tuple[dict, pl.DataFrame]:
    x, y = part["universe_return"].to_numpy(), part["portfolio_return"].to_numpy()
    paired = np.isfinite(x) & np.isfinite(y)
    n = int(paired.sum())
    intercept = beta = annual = condition = None
    annual_status = "unavailable_fit"
    conditioning_x = x[paired]
    coordinate_policy = "supplied_units"
    if n == 0:
        status = "no_joint_observations"
    elif n < 2:
        status = "insufficient_observations"
    elif np.all(x[paired] == x[paired][0]):
        status = "constant_benchmark"
    else:
        x_scale = float(np.max(np.abs(x[paired])))
        y_scale = float(np.max(np.abs(y[paired]))) or 1.0
        conditioning_x = x[paired] / x_scale
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            scaled_intercept, scaled_beta = alpha_beta(
                y[paired] / y_scale, conditioning_x, periods_per_year=1
            )
            intercept = float(scaled_intercept * y_scale)
            beta = float(
                (scaled_beta * y_scale) / x_scale
                if abs(scaled_beta) <= 1
                else scaled_beta * (y_scale / x_scale)
            )
        coordinate_policy = "unit_max_abs"
        if not np.isfinite([intercept, beta]).all():
            intercept = beta = None
            status = "nonfinite_fit"
        else:
            status = "defined"
            if intercept == -1:
                annual, annual_status = -1.0, "defined"
            elif intercept < -1:
                annual_status = "negative_gross_intercept"
            else:
                with np.errstate(over="ignore", invalid="ignore"):
                    compounded = float(np.expm1(np.log1p(intercept) * scale))
                if np.isfinite(compounded):
                    annual, annual_status = compounded, "defined"
                else:
                    annual_status = "annualization_overflow"
    if n:
        design = np.column_stack([np.ones(n), conditioning_x])
        rank = int(np.linalg.matrix_rank(design))
        observed_condition = float(np.linalg.cond(design))
        if np.isfinite(observed_condition):
            condition = observed_condition
    else:
        rank = 0
    fitted = np.full(len(x), np.nan)
    residual = np.full(len(x), np.nan)
    if intercept is not None and beta is not None:
        with np.errstate(over="ignore", invalid="ignore"):
            fitted_scaled = scaled_intercept + scaled_beta * conditioning_x
            fitted[paired] = fitted_scaled * y_scale
            residual[paired] = (y[paired] / y_scale - fitted_scaled) * y_scale
        fitted[~np.isfinite(fitted)] = np.nan
        residual[~np.isfinite(residual)] = np.nan
    observations = part.with_columns(
        pl.Series("paired", paired),
        pl.Series("fitted_return", fitted, nan_to_null=True),
        pl.Series("residual", residual, nan_to_null=True),
    )
    return {
        "return_column": part["return_column"][0] if len(part) else None,
        "annualization_factor": scale,
        "alpha_per_period": intercept,
        "annual_alpha": annual,
        "beta": beta,
        "n_dates": len(part),
        "n_obs": n,
        "n_portfolio_dates": int(np.isfinite(y).sum()),
        "n_universe_dates": int(np.isfinite(x).sum()),
        "design_rank": rank,
        "design_condition_number": condition,
        "fit_coordinate_policy": coordinate_policy,
        "status": status,
        "annualization_status": annual_status,
    }, observations


def compute_factor_alpha_beta(
    data: pl.DataFrame | pd.DataFrame,
    *,
    return_columns: Sequence[str],
    annualization_factors: Mapping[str, float],
    returns: pl.DataFrame | pd.DataFrame | pd.Series | None = None,
    demeaned: bool = True,
    group_adjust: bool = False,
    equal_weight: bool = False,
    factor_col: str = "factor",
    date_col: str = "date",
    asset_col: str = "asset",
    group_col: str = "group",
) -> FactorAlphaBetaResult:
    """Regress dated factor returns on equal-asset same-universe returns.

    Reuse ``compute_factor_returns`` for all weighting options. The benchmark is
    the finite arithmetic asset mean of each supplied horizon in ``data``, not
    an external market index or group-neutral benchmark. Missing future outcomes
    do not change weights. With ``returns`` supplied, use its unique date grid
    and wide horizon columns (or one indexed Pandas Series), align by date and
    ignore weighting options. Unmatched dates remain visible with zero benchmark
    counts and null benchmark returns. Inputs are never mutated.

    ``annualization_factors`` must explicitly name each selected return column
    with a positive finite scale. Annual alpha is ``(1+intercept)**scale-1``;
    beta is dimensionless. For the legacy fixed252-day convention use252 for1D
    and50.4 for5D, only when those durations are explicitly known. Column labels
    alone establish neither duration nor calendar. Generic ``alpha_beta`` keeps
    its existing linear annualization and defaults.

    Regress only finite paired portfolio/benchmark dates using the existing OLS
    kernel in unit-max-absolute coordinates, then restore coefficient units.
    Rank/condition diagnostics describe that scaled design, as recorded in
    fit_coordinate_policy. Finite mean overflow uses the shared stable sample
    statistics without dropping dates. Fewer than two pairs or constant benchmark yields null coefficients
    with a status. Unlike released Alphalens0.4.6, all-missing portfolio dates
    stay null rather than zero, and a constant-zero benchmark does not invent
    identified beta0. Zero signal uses the existing explicit zero-exposure policy.
    Negative gross intercepts and overflowing compound alpha are unavailable,
    with a separate annualization status; finite period alpha/beta are retained.
    This computes no t-statistic, wealth path, holdings or report panels.
    """
    _weight_flags(demeaned=demeaned, group_adjust=group_adjust, equal_weight=equal_weight)
    values, columns = _signal_inputs(
        data,
        return_columns,
        factor_col,
        date_col,
        asset_col,
        group_col if group_adjust and returns is None else None,
    )
    if not isinstance(annualization_factors, Mapping) or set(annualization_factors) != set(columns):
        raise ValueError("annualization_factors must name exactly the selected return columns")
    scales = {}
    for column in columns:
        value = annualization_factors[column]
        if isinstance(value, bool | np.bool_) or not isinstance(value, Real):
            raise ValueError("annualization_factors must contain positive finite numbers")
        try:
            scale = float(value)
        except (OverflowError, ValueError) as error:
            raise ValueError(
                "annualization_factors must contain positive finite numbers"
            ) from error
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("annualization_factors must contain positive finite Float64 numbers")
        scales[column] = scale
    if returns is None:
        portfolio = compute_factor_returns(
            values,
            return_columns=columns,
            demeaned=demeaned,
            group_adjust=group_adjust,
            equal_weight=equal_weight,
            factor_col=factor_col,
            date_col=date_col,
            asset_col=asset_col,
            group_col=group_col,
        ).rename({"weighted_return": "portfolio_return", "n_obs": "n_portfolio"})
    else:
        portfolio = _provided_returns(returns, columns, date_col)
    statistics, observations = [], []
    for column in columns:
        benchmark = _universe_means(values, column, date_col)
        if returns is None:
            part = portfolio.filter(pl.col("return_column") == column).select(
                "date", "portfolio_return", "n_portfolio"
            )
        else:
            part = portfolio.select(
                "date",
                pl.when(pl.col(column).cast(pl.Float64).is_finite())
                .then(pl.col(column).cast(pl.Float64))
                .otherwise(None)
                .alias("portfolio_return"),
            ).with_columns(
                pl.col("portfolio_return").is_not_null().cast(pl.UInt32).alias("n_portfolio")
            )
        part, benchmark = _align_datetime_key_resolution(part, benchmark, "date")
        if part.schema["date"] != benchmark.schema["date"]:
            raise ValueError(
                "provided factor returns and input dates require matching types/timezones"
            )
        part = (
            part.join(benchmark, on="date", how="left")
            .sort("date")
            .with_columns(pl.col("n_universe").fill_null(0), pl.lit(column).alias("return_column"))
        )
        row, sample = _fit(part, scales[column])
        row["return_column"] = column
        statistics.append(row)
        observations.append(sample)
    stats = pl.DataFrame(
        statistics,
        schema_overrides=dict.fromkeys(
            [
                "annualization_factor",
                "alpha_per_period",
                "annual_alpha",
                "beta",
                "design_condition_number",
            ],
            pl.Float64,
        ),
    )
    metadata = {
        "return_columns": columns,
        "annualization": "explicit_compounded_intercept",
        "benchmark": "equal_asset_mean_of_same_supplied_universe",
        "sample_policy": "finite_joint_dates_no_missing_outcome_weight_reallocation",
        "portfolio_source": "provided" if returns is not None else "factor_weights",
        "weight_options": None
        if returns is not None
        else {
            "demeaned": bool(demeaned),
            "group_adjust": bool(group_adjust),
            "equal_weight": bool(equal_weight),
        },
        "n_portfolio_units": "finite supplied dates"
        if returns is not None
        else "finite weighted asset contributions",
        "zero_weight_policy": "zero",
    }
    return FactorAlphaBetaResult(
        stats.write_ipc(None).getvalue(),
        pl.concat(observations).write_ipc(None).getvalue(),
        json.dumps(metadata, allow_nan=False),
    )
