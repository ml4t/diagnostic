"""Supplied beginning-period NAV weights and loadings, without fitted exposures."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl

from ..portfolio_analysis.holdings import _numeric, _validate_asset_ids
from ..portfolio_analysis.inputs import date_index, date_snapshot, return_snapshot

type Frame = pl.DataFrame | pd.DataFrame
type Timing = Literal["beginning_known_at", "legacy_observed"]
type MissingLoadings = Literal["raise", "legacy_skip"]


@dataclass(frozen=True)
class HoldingsAttributionResult:
    """Owned observations and additive contributions under a declared linking method.

    Specific returns include unattributed cash unless cash contributions were
    supplied separately. Coverage reports missing gross weight for each factor;
    a skipped loading is not evidence of zero factor exposure.
    """

    exposures: pl.DataFrame
    contributions: pl.DataFrame
    cumulative: pl.DataFrame
    coverage: pl.DataFrame
    factor_names: tuple[str, ...]
    cumulative_method: str
    timing: str
    cash_attributed: bool

    def to_dataframe(self) -> pl.DataFrame:
        """Return an independent period contribution table."""
        return self.contributions.clone()

    def to_dict(self) -> dict[str, object]:
        """Return owned tables and declared methods, without claiming JSON encoding."""
        return {
            "exposures": self.exposures.to_dicts(),
            "contributions": self.contributions.to_dicts(),
            "cumulative": self.cumulative.to_dicts(),
            "coverage": self.coverage.to_dicts(),
            "factor_names": list(self.factor_names),
            "cumulative_method": self.cumulative_method,
            "timing": self.timing,
            "cash_attributed": self.cash_attributed,
        }


def _frame(data: Frame, *, timestamp_col: str, entity_col: str) -> pl.DataFrame:
    if isinstance(data, pl.DataFrame):
        frame = data.clone()
    elif isinstance(data, pd.DataFrame):
        frame = pl.from_pandas(data.copy(deep=True))
    else:
        raise TypeError("attribution inputs must be pandas or Polars DataFrames")
    mapping = {}
    for given, canonical in ((timestamp_col, "timestamp"), (entity_col, "asset")):
        if given != canonical and given in frame.columns:
            if canonical in frame.columns:
                raise ValueError(f"conflicting {given}/{canonical} column declarations")
            mapping[given] = canonical
    return frame.rename(mapping)


def _required(frame: pl.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")


def _locations(frame: pl.DataFrame, dates: pl.Series, name: str) -> np.ndarray:
    _required(frame, ["timestamp"], name)
    supplied = date_snapshot(
        frame["timestamp"].unique(), frame["timestamp"].n_unique(), f"{name} dates", ordered=False
    )
    wanted, actual = date_index(dates), date_index(supplied)
    if actual.tz != wanted.tz:
        raise ValueError(f"{name} dates must use the return timezone")
    return wanted.get_indexer(date_index(frame["timestamp"]))


def _aligned(frame: pl.DataFrame, dates: pl.Series, name: str) -> pl.DataFrame:
    _required(frame, ["timestamp"], name)
    supplied = date_snapshot(frame["timestamp"], frame.height, name, ordered=False)
    actual, expected = date_index(supplied), date_index(dates)
    if actual.tz != expected.tz:
        raise ValueError(f"{name} dates must use the return timezone")
    locations = actual.get_indexer(expected)
    if np.any(locations < 0):
        raise ValueError(f"{name} must cover every return observation")
    return frame[locations]


def _available(frame: pl.DataFrame, starts: pl.Series, rows: np.ndarray, name: str) -> None:
    if frame.height == 0:
        return
    _required(frame, ["known_at"], name)
    known = frame["known_at"]
    if not isinstance(known.dtype, pl.Datetime) or known.null_count():
        raise ValueError(f"{name} known_at requires nonmissing Datetime instants")
    observed, beginning = date_index(known), date_index(starts.gather(rows))
    if observed.tz != beginning.tz or np.any(observed > beginning):
        raise ValueError(
            f"{name} known_at must be available at the beginning of its return interval"
        )


def _exposures(
    weights: pl.DataFrame,
    loadings: pl.DataFrame,
    dates: pl.Series,
    names: tuple[str, ...],
    *,
    starts: pl.Series | None,
    timing: Timing,
    missing_loadings: MissingLoadings,
    observed_snapshot_dates: pl.Series | pd.DatetimeIndex | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    for frame, label in ((weights, "weights"), (loadings, "loadings")):
        _required(frame, ["timestamp", "asset"], label)
        _validate_asset_ids(frame["asset"])
        if frame.select(pl.struct("timestamp", "asset").is_duplicated().any()).item():
            raise ValueError(f"{label} timestamp/asset keys must be unique")
    weights = _numeric(weights, "weight")
    for name in names:
        _required(loadings, [name], "loadings")
        loadings = _numeric(loadings, name, nullable=True)
    locations = _locations(weights, dates, "weights")
    if timing == "beginning_known_at":
        supplied_dates = (
            weights["timestamp"].unique()
            if observed_snapshot_dates is None
            else observed_snapshot_dates
        )
        snapshots = date_snapshot(
            supplied_dates, len(supplied_dates), "weight snapshot dates", ordered=False
        )
        actual, expected = date_index(snapshots), date_index(dates)
        if actual.tz != expected.tz or np.any(actual.get_indexer(expected) < 0):
            raise ValueError("weight snapshot dates must cover every return observation")
    load_locations = _locations(loadings, dates, "loadings")
    weights = weights.filter(pl.Series(locations >= 0))
    locations = locations[locations >= 0]
    loadings = loadings.filter(pl.Series(load_locations >= 0))
    load_locations = load_locations[load_locations >= 0]
    if timing == "beginning_known_at":
        assert starts is not None
        _available(weights, starts, locations, "weights")
        _available(loadings, starts, load_locations, "loadings")
    left = pd.MultiIndex.from_arrays([date_index(weights["timestamp"]), weights["asset"].to_list()])
    right = pd.MultiIndex.from_arrays(
        [date_index(loadings["timestamp"]), loadings["asset"].to_list()]
    )
    selected = right.get_indexer(left)
    values = weights["weight"].to_numpy()
    matrix = np.full((len(values), len(names)), np.nan)
    matched = selected >= 0
    matrix[matched] = loadings.select(names).to_numpy()[selected[matched]]
    missing = ~np.isfinite(matrix)
    if missing_loadings == "raise" and np.any(missing & (values[:, None] != 0)):
        raise ValueError("loadings do not cover every nonzero asset/factor weight")
    known = np.where(missing, 0.0, matrix)
    exposures = np.zeros((len(dates), len(names)))
    uncovered = np.zeros_like(exposures)
    gross = np.zeros(len(dates))
    np.add.at(exposures, locations, values[:, None] * known)
    np.add.at(uncovered, locations, np.abs(values[:, None]) * missing)
    np.add.at(gross, locations, np.abs(values))
    timeline = {"timestamp": dates.rename("timestamp")}
    exposure_frame = pl.DataFrame({**timeline, **dict(zip(names, exposures.T, strict=True))})
    coverage = pl.DataFrame(
        {
            "timestamp": dates.gather(np.repeat(np.arange(len(dates)), len(names))).rename(
                "timestamp"
            ),
            "factor": list(names) * len(dates),
            "gross_weight": np.repeat(gross, len(names)),
            "covered_gross_weight": (gross[:, None] - uncovered).ravel(),
            "uncovered_gross_weight": uncovered.ravel(),
        }
    )
    return exposure_frame, coverage


def compute_holdings_attribution(
    returns: Frame,
    weights: Frame,
    factor_returns: Frame,
    factor_loadings: Frame,
    *,
    factor_names: Sequence[str] | None = None,
    complete_snapshots: bool = False,
    cash_contributions: Frame | None = None,
    cumulative_method: Literal["wealth_linked", "arithmetic"] = "wealth_linked",
    timing: Timing = "beginning_known_at",
    missing_loadings: MissingLoadings = "raise",
    timestamp_col: str = "timestamp",
    entity_col: str = "asset",
    observed_snapshot_dates: pl.Series | pd.DatetimeIndex | None = None,
) -> HoldingsAttributionResult:
    """Attribute supplied fractional NAV weights without estimating any betas.

    Returns have timestamp/return and, for native timing, period_start columns.
    Weights have timestamp/asset/weight; loadings add named factor columns. Native
    weights and loadings require known_at instants no later than period_start.
    Timestamp keys identify the return interval; they are not inferred sessions.
    The explicit complete snapshot declaration makes absent assets zero weights.
    Every return needs an observed snapshot date. Supply observed_snapshot_dates
    from an actual account timeline for all-cash dates with no asset rows.
    Dollar positions must be normalized using declared matching NAV beforehand.

    Named legacy timing consumes observation labels without a causal availability
    claim. Named legacy missingness skips missing factor loadings while retaining
    uncovered gross weight in coverage. Neither is the native default.
    """
    if complete_snapshots is not True:
        raise ValueError("complete_snapshots=True must declare complete supplied weight snapshots")
    if cumulative_method not in ("wealth_linked", "arithmetic"):
        raise ValueError("cumulative_method must be wealth_linked or arithmetic")
    if timing not in ("beginning_known_at", "legacy_observed"):
        raise ValueError("unknown holdings attribution timing policy")
    if missing_loadings not in ("raise", "legacy_skip"):
        raise ValueError("missing_loadings must be raise or legacy_skip")
    if timestamp_col == entity_col:
        raise ValueError("timestamp_col and entity_col must identify different fields")
    frames = [
        _frame(frame, timestamp_col=timestamp_col, entity_col=entity_col)
        for frame in (returns, weights, factor_returns, factor_loadings)
    ]
    realized, owned_weights, factors, loadings = frames
    _required(realized, ["timestamp", "return"], "returns")
    dates = date_snapshot(realized["timestamp"], realized.height, "returns dates")
    total = return_snapshot(realized["return"], "returns")
    names = (
        tuple(factor_names)
        if factor_names is not None
        else tuple(name for name in factors.columns if name != "timestamp")
    )
    reserved = {
        "timestamp",
        "total_returns",
        "common_returns",
        "specific_returns",
        "cash_returns",
        "tilt_returns",
        "timing_returns",
    }
    if (
        not names
        or len(set(names)) != len(names)
        or any(not isinstance(name, str) or not name or name in reserved for name in names)
    ):
        raise ValueError(
            "factor_names must be unique nonempty factor names, not reserved output fields"
        )
    factors = _aligned(factors, dates, "factor returns")
    for name in names:
        _required(factors, [name], "factor returns")
        factors = _numeric(factors, name)
    starts = None
    if timing == "beginning_known_at":
        _required(realized, ["period_start"], "returns")
        starts = realized["period_start"]
        if (
            not isinstance(dates.dtype, pl.Datetime)
            or not isinstance(starts.dtype, pl.Datetime)
            or starts.null_count()
        ):
            raise ValueError("beginning_known_at timing requires Datetime interval instants")
        beginnings, endings = date_index(starts), date_index(dates)
        if beginnings.tz != endings.tz or np.any(beginnings >= endings):
            raise ValueError("period_start must precede its return timestamp on the same clock")
        if len(beginnings) > 1 and np.any(beginnings[1:] < endings[:-1]):
            raise ValueError("realized return intervals must not overlap for account attribution")
    exposures, coverage = _exposures(
        owned_weights,
        loadings,
        dates,
        names,
        starts=starts,
        timing=timing,
        missing_loadings=missing_loadings,
        observed_snapshot_dates=observed_snapshot_dates,
    )
    exposure_values = exposures.select(names).to_numpy()
    factor_values = factors.select(names).to_numpy()
    contributions = exposure_values * factor_values
    common = contributions.sum(axis=1)
    tilt = (exposure_values.mean(axis=0) * factor_values).sum(axis=1)
    columns = {
        **dict(zip(names, contributions.T, strict=True)),
        "total_returns": total,
        "common_returns": common,
        "specific_returns": total - common,
        "tilt_returns": tilt,
        "timing_returns": common - tilt,
    }
    if cash_contributions is not None:
        cash = _frame(cash_contributions, timestamp_col=timestamp_col, entity_col=entity_col)
        _required(cash, ["cash_contribution"], "cash contributions")
        cash = _numeric(_aligned(cash, dates, "cash contributions"), "cash_contribution")
        cash_values = cash["cash_contribution"].to_numpy()
        columns["cash_returns"] = cash_values
        columns["specific_returns"] = total - common - cash_values
    prior_wealth = np.r_[1.0, np.cumprod(1 + total[:-1])]
    multiplier = prior_wealth if cumulative_method == "wealth_linked" else np.ones_like(total)
    return HoldingsAttributionResult(
        exposures=exposures,
        contributions=pl.DataFrame({"timestamp": dates.rename("timestamp"), **columns}),
        cumulative=pl.DataFrame(
            {
                "timestamp": dates.rename("timestamp"),
                **{name: np.cumsum(multiplier * values) for name, values in columns.items()},
            }
        ),
        coverage=coverage,
        factor_names=names,
        cumulative_method=cumulative_method,
        timing=timing,
        cash_attributed=cash_contributions is not None,
    )
