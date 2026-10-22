"""Pyfolio return statistics, dated events and IID cones over shared native kernels.

The familiar IID stream uses RandomState, unlike the native Generator default.
Missing sampled returns are neutral factors in the existing product kernel.
Zero starting_value denotes net cumulative return, not a zero-capital account.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

from ml4t.diagnostic.compat import empyrical as ep
from ml4t.diagnostic.compat._returns import ReturnInput, as_array
from ml4t.diagnostic.evaluation.portfolio_analysis.inputs import positive_integer
from ml4t.diagnostic.evaluation.portfolio_analysis.metrics import gaussian_value_at_risk
from ml4t.diagnostic.evaluation.portfolio_analysis.regimes import _PERIOD_DATES
from ml4t.diagnostic.evaluation.portfolio_analysis.wealth_cones import (
    _iid_indices,
    _sample_indices,
    _wealth_paths,
)

__all__ = [
    "value_at_risk",
    "gross_lev",
    "perf_stats",
    "calc_bootstrap",
    "calc_distribution_stats",
    "perf_stats_bootstrap",
    "simulate_paths",
    "summarize_paths",
    "forecast_cone_bootstrap",
    "extract_interesting_date_ranges",
    "rolling_volatility",
    "rolling_sharpe",
    "rolling_beta",
    "gen_drawdown_table",
]


def _rolling_complete(returns: pd.Series, window: int, metric: str, scale: int) -> pd.Series:
    width = positive_integer(window, "rolling window")
    scale = positive_integer(scale, "periods_per_year")
    if not isinstance(returns, pd.Series):
        raise TypeError("rolling returns must be a pandas Series")
    if len(returns) < width:
        return pd.Series(np.nan, index=returns.index, dtype=float)
    values = getattr(ep, metric)(returns, window=width, annualization=scale).reindex(returns.index)
    # The familiar Pyfolio path requires every row, unlike Empyrical's NaN-omitting windows.
    return values.where(returns.rolling(width).count() == width)


def rolling_volatility(
    returns: pd.Series, rolling_vol_window: int, *, periods_per_year: int = 252
) -> pd.Series:
    """Annualized sample dispersion on full observed row windows, with padded warmup."""
    return _rolling_complete(
        returns, rolling_vol_window, "roll_annual_volatility", periods_per_year
    )


def rolling_sharpe(
    returns: pd.Series, rolling_sharpe_window: int, *, periods_per_year: int = 252
) -> pd.Series:
    """Zero-periodic-RF Sharpe with original full-window missingness and explicit scale."""
    return _rolling_complete(returns, rolling_sharpe_window, "roll_sharpe_ratio", periods_per_year)


def rolling_beta(
    returns: pd.Series, factor_returns: pd.Series | pd.DataFrame, rolling_window: int = 126
) -> pd.Series | pd.DataFrame:
    """Retain Pyfolio's inclusive endpoints: window126 selects127 observed rows."""
    width = positive_integer(rolling_window, "rolling_window") + 1
    if isinstance(factor_returns, pd.DataFrame):
        return factor_returns.apply(lambda factor: rolling_beta(returns, factor, rolling_window))
    if not isinstance(returns, pd.Series) or not isinstance(factor_returns, pd.Series):
        raise TypeError("rolling beta requires pandas Series or benchmark columns")
    if not returns.index.is_unique or not factor_returns.index.is_unique:
        raise ValueError("rolling beta requires unique observation keys")
    if len(returns) < width:
        return pd.Series(np.nan, index=returns.index, dtype=float)
    factor = factor_returns.reindex(returns.index)
    return ep.roll_beta(returns, factor, window=width).reindex(returns.index)


def gen_drawdown_table(returns: pd.Series, top: int = 10) -> pd.DataFrame:
    """Project shared initial-capital episodes; do not invent an initial timestamp.

    Depth is percent. Duration retains the original inclusive civil-business-date
    count only when both peak and recovery dates exist, not exchange sessions.
    Empty slots are padded as in the familiar table. A monotone gain has no episode.
    """
    from ml4t.diagnostic.evaluation.portfolio_analysis.drawdowns import _identify_drawdown_periods

    from .txn import _dated

    _dated(returns, "returns")
    if isinstance(top, bool | np.bool_) or not isinstance(top, int | np.integer) or top < 0:
        raise ValueError("top must be a nonnegative integer")
    curve = ep.drawdown_series(returns)
    episodes = sorted(
        _identify_drawdown_periods(as_array(curve), returns.index, 0.0), key=lambda p: p.depth
    )
    table = pd.DataFrame(
        np.nan,
        index=range(top),
        columns=["Net drawdown in %", "Peak date", "Valley date", "Recovery date", "Duration"],
    ).astype(object)
    for row, episode in enumerate(episodes[:top]):
        peak, valley, recovery = [
            pd.NaT if value is None else pd.Timestamp(value)
            for value in (episode.peak_date, episode.valley_date, episode.recovery_date)
        ]
        table.loc[row] = [
            -100 * episode.depth,
            peak,
            valley,
            recovery,
            len(pd.bdate_range(peak, recovery))
            if pd.notna(peak) and pd.notna(recovery)
            else np.nan,
        ]
    for name in ("Peak date", "Valley date", "Recovery date"):
        table[name] = pd.to_datetime(table[name], utc=returns.index.tz is not None)
        if returns.index.tz is not None:
            table[name] = table[name].dt.tz_convert(returns.index.tz)
    table.attrs["diagnostic"] = {
        "depth_units": "percent",
        "initial_capital_peak_timestamp": "unavailable",
        "duration_units": "inclusive civil Monday-Friday dates, not exchange sessions",
        "initial_capital_included": True,
        "padding_is_unavailable_not_zero_risk": True,
    }
    return table


_STAT_NAMES = {
    "annual_return": "Annual return",
    "cum_returns_final": "Cumulative returns",
    "annual_volatility": "Annual volatility",
    "sharpe_ratio": "Sharpe ratio",
    "calmar_ratio": "Calmar ratio",
    "stability_of_timeseries": "Stability",
    "max_drawdown": "Max drawdown",
    "omega_ratio": "Omega ratio",
    "sortino_ratio": "Sortino ratio",
    "skew": "Skew",
    "kurtosis": "Kurtosis",
    "tail_ratio": "Tail ratio",
    "value_at_risk": "Daily value at risk",
}


def value_at_risk(returns: ReturnInput, period: str | None = None, sigma: float = 2.0) -> float:
    """Use the shared Gaussian periodic-return estimate, not empirical VaR."""
    supplied = ep.aggregate_returns(returns, period) if period is not None else returns
    return gaussian_value_at_risk(supplied, sigma=sigma)


def gross_lev(positions: pd.DataFrame) -> pd.Series:
    """Adapt shared gross books and explicit legacy signed/zero-NAV division."""
    from ml4t.diagnostic.evaluation.portfolio_analysis.transaction_metrics import cash_fraction_expr

    from .txn import _books

    if "cash" not in positions:
        raise KeyError("cash")
    gross, nav = _books(positions)
    values = (
        pl.DataFrame({"gross": gross.to_numpy(), "nav": nav.to_numpy()})
        .select(cash_fraction_expr(pl.col("gross"), pl.col("nav"), positive_finite=False))
        .to_series()
        .to_numpy()
    )
    return pd.Series(values, index=positions.index.copy())


def _statistic(name: str, returns: ReturnInput, periods_per_year: int) -> float:
    if name == "value_at_risk":
        return value_at_risk(returns)
    if name in {"skew", "kurtosis"}:
        return float(getattr(stats, name)(as_array(returns)))
    function = getattr(ep, name)
    if name in {
        "annual_return",
        "annual_volatility",
        "sharpe_ratio",
        "calmar_ratio",
        "sortino_ratio",
    }:
        return float(function(returns, annualization=periods_per_year))
    return float(function(returns))


def perf_stats(
    returns: ReturnInput,
    factor_returns: ReturnInput | None = None,
    positions: pd.DataFrame | None = None,
    transactions: pd.DataFrame | None = None,
    turnover_denom: str = "AGB",
    *,
    periods_per_year: int = 252,
) -> pd.Series:
    """Project the actual Pyfolio table through existing metric/account kernels.

    NaN behavior is per metric: CAGR counts elapsed rows, deviations omit NaNs,
    while original SciPy skew/kurtosis propagate them. Gaussian daily VaR is
    mean minus two sample deviations, not Empyrical's quantile. Native summary
    definitions and annual-rate APIs remain unchanged. An explicit scale is an
    extension to the legacy table's fixed252default.
    """
    scale = positive_integer(periods_per_year, "periods_per_year")
    if as_array(returns).ndim != 1:
        raise ValueError("perf_stats requires one return series")
    output = {label: _statistic(name, returns, scale) for name, label in _STAT_NAMES.items()}
    if positions is not None and not positions.empty:
        output["Gross leverage"] = gross_lev(positions).mean()
        if transactions is not None and not transactions.empty:
            from .txn import get_turnover

            output["Daily turnover"] = get_turnover(positions, transactions, turnover_denom).mean()
    if factor_returns is not None:
        output["Alpha"] = float(ep.alpha(returns, factor_returns, annualization=scale))
        output["Beta"] = float(ep.beta(returns, factor_returns))
    return pd.Series(output, dtype=float)


def _bootstrap_values(
    function: Callable,
    returns: pd.Series,
    args: tuple,
    kwargs: dict,
    factor_returns: pd.Series | None,
    n_samples: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    if not isinstance(returns, pd.Series) or returns.empty:
        raise ValueError("bootstrap returns require a nonempty pandas Series")
    if factor_returns is not None and (
        not isinstance(factor_returns, pd.Series) or len(factor_returns) != len(returns)
    ):
        raise ValueError("bootstrap benchmark must be a same-length pandas Series")
    output = np.empty(n_samples)
    for row in range(n_samples):
        selected = _iid_indices(len(returns), len(returns), 1, rng)[0]
        sample = returns.iloc[selected].reset_index(drop=True)
        if factor_returns is None:
            output[row] = function(sample, *args, **kwargs)
        else:
            factor = factor_returns.iloc[selected].reset_index(drop=True)
            output[row] = function(sample, factor, *args, **kwargs)
    return output


def calc_bootstrap(
    function: Callable[..., float], returns: pd.Series, *args: Any, **kwargs: Any
) -> np.ndarray:
    """Resample paired positional rows with a local IID stream and O(n) storage.

    random_seed is an explicit extension; global NumPy state is never changed.
    Original date labels reset before the statistic call, as in the reference.
    This is an IID sampling distribution, not native stationary/HAC inference.
    """
    count = positive_integer(kwargs.pop("n_samples", 1000), "n_samples")
    factor = kwargs.pop("factor_returns", None)
    rng = np.random.RandomState(kwargs.pop("random_seed", None))
    return _bootstrap_values(function, returns, args, kwargs, factor, count, rng)


def calc_distribution_stats(values: np.ndarray | pd.Series) -> pd.Series:
    """Preserve the original mixed NumPy/pandas reductions and population deviation."""
    return pd.Series(
        {
            "mean": np.mean(values),
            "median": np.median(values),
            "std": np.std(values),
            "5%": np.percentile(values, 5),
            "25%": np.percentile(values, 25),
            "75%": np.percentile(values, 75),
            "95%": np.percentile(values, 95),
            "IQR": np.subtract.reduce(np.percentile(values, [75, 25])),
        }
    )


def perf_stats_bootstrap(
    returns: pd.Series,
    factor_returns: pd.Series | None = None,
    return_stats: bool = True,
    *,
    n_samples: int = 1000,
    random_seed: int | None = None,
    periods_per_year: int = 252,
    positions: pd.DataFrame | None = None,
    transactions: pd.DataFrame | None = None,
    turnover_denom: str = "AGB",
) -> pd.DataFrame:
    """Return actual IID draws or their mean/median/5/95percent sampling summaries.

    As in Pyfolio, this bootstrap covers return/benchmark metrics, not changing
    positions, fills or turnover. The legacy function silently ignores kwargs;
    this adapter honors explicit sample count, local seed and annualization.
    All metrics share one advancing local stream, not repeated reseeding.
    """
    count = positive_integer(n_samples, "n_samples")
    scale = positive_integer(periods_per_year, "periods_per_year")
    if not isinstance(return_stats, bool | np.bool_):
        raise TypeError("return_stats must be boolean")
    _ = positions, transactions, turnover_denom
    rng = np.random.RandomState(random_seed)
    draws = {}
    for name, label in _STAT_NAMES.items():
        draws[label] = _bootstrap_values(
            lambda sample, name=name: _statistic(name, sample, scale),
            returns,
            (),
            {},
            None,
            count,
            rng,
        )
    if factor_returns is not None:
        for label, function in [("Alpha", ep.alpha), ("Beta", ep.beta)]:
            options = {"annualization": scale} if label == "Alpha" else {}
            draws[label] = _bootstrap_values(
                function, returns, (), options, factor_returns, count, rng
            )
    result = pd.DataFrame(draws)
    result.attrs["diagnostic"] = {
        "method": "iid",
        "n_samples": count,
        "seed": random_seed,
        "periods_per_year": scale,
        "input_rows": len(returns),
        "pairing": "positional reset labels",
        "positions_and_transactions": "not bootstrapped",
    }
    if return_stats:
        summary = result.apply(calc_distribution_stats).T[["mean", "median", "5%", "95%"]]
        summary.attrs = result.attrs.copy()
        return summary
    return result


def simulate_paths(
    is_returns: pd.Series,
    num_days: int,
    starting_value: float = 1,
    num_samples: int = 1000,
    random_seed: int | None = None,
) -> np.ndarray:
    """Return raw IID draws with the familiar local seeded stream and sample shape.

    num_days counts output observations, not exchange sessions. starting_value
    has no role in raw return sampling; summarize_paths applies that convention.
    """
    if not isinstance(is_returns, pd.Series):
        raise TypeError("is_returns must be a one-dimensional pandas Series")
    indices = _sample_indices(
        len(is_returns),
        num_days,
        num_samples,
        method="iid",
        seed=random_seed,
        block_size=None,
        legacy_rng=True,
    )
    return is_returns.to_numpy(dtype=float)[indices]


def summarize_paths(
    samples: np.ndarray,
    cone_std: float | Sequence[float] = (1.0, 1.5, 2.0),
    starting_value: float = 1.0,
) -> pd.DataFrame:
    """Summarize path wealth with mean plus/minus population standard deviation.

    Numeric float columns retain input deviation order and positive/negative
    labels. These bounds are not quantiles or confidence intervals for a mean.
    Native inputs remain finite-only; this explicit legacy profile allows NaNs.
    """
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2:
        raise ValueError("samples must be a two-dimensional sample-by-step array")
    wealth = _wealth_paths(values, 1.0 if starting_value == 0 else starting_value)
    if starting_value == 0:
        wealth = wealth - 1
    means, stds = wealth.mean(axis=0), wealth.std(axis=0, ddof=0)
    levels = [cone_std] if isinstance(cone_std, int | float | np.floating) else cone_std
    result = pd.DataFrame(columns=pd.Index([], dtype="float64"))
    for level in levels:
        result[float(level)] = means + stds * level
        result[float(-level)] = means - stds * level
    result.attrs.update(
        {
            "reference_profile": "Pyfolio 0.10.0",
            "definition": "mean +/- population std, not quantiles",
            "missing_returns": "neutral compounded factor",
            "starting_value_zero": "net cumulative return",
            "horizon_unit": "observations",
        }
    )
    return result


def forecast_cone_bootstrap(
    is_returns: pd.Series,
    num_days: int,
    cone_std: float | Sequence[float] = (1.0, 1.5, 2.0),
    starting_value: float = 1,
    num_samples: int = 1000,
    random_seed: int | None = None,
) -> pd.DataFrame:
    """Compose the familiar raw IID draws and explicitly defined cone bounds."""
    return summarize_paths(
        simulate_paths(is_returns, num_days, starting_value, num_samples, random_seed),
        cone_std,
        starting_value,
    )


def extract_interesting_date_ranges(
    returns: pd.Series,
    periods: Mapping[str, tuple[date | datetime | str, date | datetime | str]] | None = None,
) -> OrderedDict[str, pd.Series]:
    """Select familiar pandas label slices, retaining actual missing observations.

    Default historical endpoints are UTC instants, as in Pyfolio. Covid's open
    end is evaluated at call time rather than fixed at module import. Explicit
    periods make the selection reproducible. Unlike Pyfolio's blanket exception
    suppression, invalid boundaries raise instead of silently hiding a period.
    Native compute_stress additionally retains uncovered periods and coverage.
    """
    if not isinstance(returns, pd.Series) or not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must be a pandas Series with a DatetimeIndex")
    windows = periods
    if windows is None:
        windows = {
            name: (
                pd.Timestamp(start, tz="UTC"),
                pd.Timestamp.now(tz="UTC") if end is None else pd.Timestamp(end, tz="UTC"),
            )
            for name, (start, end) in _PERIOD_DATES.items()
        }
    result = OrderedDict()
    source = returns.copy(deep=True)
    source.index = source.index.map(pd.Timestamp)
    for name, (start, end) in windows.items():
        sample = source.loc[start:end].copy(deep=True)
        if not sample.empty:
            result[name] = sample
    return result
