"""Observation-aligned signal event paths, separate from abnormal-return models."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from io import BytesIO
from numbers import Integral
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl

from ml4t.diagnostic.evaluation.portfolio_analysis.metrics import _safe_cumprod
from ml4t.diagnostic.signal.quantile import _sample_return_statistics
from ml4t.diagnostic.utils.validation import _validate_observation_keys

_TABLES = ("paths", "observations", "statistics", "event_counts")


@dataclass(frozen=True)
class AlignedEventResult:
    """Owned event paths, effective observations, sample statistics and counts.

    Offsets count rows in the supplied value clock, not days or sessions.
    Levels are not rebased to event zero. Standard deviations describe the
    effective paths, not standard errors or confidence intervals.
    """

    _tables: tuple[bytes, ...] = field(repr=False)
    _metadata: str = field(repr=False)

    @property
    def paths(self) -> pl.DataFrame:
        """Inspect each event/offset, source timestamp, value and missing status."""
        return pl.read_ipc(BytesIO(self._tables[0]))

    @property
    def observations(self) -> pl.DataFrame:
        """Inspect the actual event or date/group-mean sample used for statistics."""
        return pl.read_ipc(BytesIO(self._tables[1]))

    @property
    def statistics(self) -> pl.DataFrame:
        """Inspect bucket/group/offset means, sample deviations and finite counts."""
        return pl.read_ipc(BytesIO(self._tables[2]))

    @property
    def event_counts(self) -> pl.DataFrame:
        """Inspect event counts at their supplied timestamps, before path clipping."""
        return pl.read_ipc(BytesIO(self._tables[3]))

    @property
    def metadata(self) -> dict:
        """Return independent units, clock and aggregation declarations."""
        return json.loads(self._metadata)

    def to_json(self) -> str:
        """Serialize strict JSON with typed Arrow snapshots, not rounded tables."""
        return json.dumps(
            {
                "format_version": 1,
                "tables": {
                    name: base64.b64encode(payload).decode("ascii")
                    for name, payload in zip(_TABLES, self._tables, strict=True)
                },
                "metadata": self.metadata,
            },
            allow_nan=False,
        )

    @classmethod
    def from_json(cls, value: str) -> AlignedEventResult:
        """Restore the supported format while preserving timestamp precision."""
        parsed = json.loads(value)
        if parsed.get("format_version") != 1 or set(parsed.get("tables", {})) != set(_TABLES):
            raise ValueError("unsupported aligned event result format")
        tables = tuple(base64.b64decode(parsed["tables"][name], validate=True) for name in _TABLES)
        for table in tables:
            pl.read_ipc(BytesIO(table))
        return cls(tables, json.dumps(parsed["metadata"], allow_nan=False))


def _event_frame(
    events: pl.DataFrame | pd.DataFrame | pd.Series, date_col: str, asset_col: str
) -> pd.DataFrame:
    if isinstance(events, pl.DataFrame):
        _validate_observation_keys(events, [date_col, asset_col], "events", unique=False)
        frame = events.to_pandas()
    elif isinstance(events, pd.Series):
        frame = events.rename("factor").to_frame().reset_index()
    elif isinstance(events, pd.DataFrame):
        frame = (
            events.reset_index()
            if isinstance(events.index, pd.MultiIndex)
            else events.copy(deep=True)
        )
    else:
        raise TypeError("events must be a Pandas/Polars frame or indexed Series")
    if frame.empty or not {date_col, asset_col}.issubset(frame.columns):
        raise ValueError("events must contain nonempty date and asset keys")
    if frame[[date_col, asset_col]].isna().any().any():
        raise ValueError("event date and asset keys must be nonmissing")
    if not all(isinstance(v, (date, datetime, np.datetime64)) for v in frame[date_col]):
        raise TypeError("event dates must contain date or datetime values")
    frame = frame.rename(columns={date_col: "event_date", asset_col: "asset"})
    frame["event_date"] = pd.DatetimeIndex(frame["event_date"]).as_unit("ns")
    if "factor_quantile" in frame:
        if "quantile" in frame:
            raise ValueError("ambiguous quantile and factor_quantile columns")
        frame = frame.rename(columns={"factor_quantile": "quantile"})
    if "quantile" not in frame:
        frame["quantile"] = 1
    buckets = frame["quantile"]
    if (
        pd.api.types.is_bool_dtype(buckets)
        or not pd.api.types.is_numeric_dtype(buckets)
        or not (buckets.notna() & (buckets >= 1) & (buckets == np.floor(buckets))).all()
    ):
        raise ValueError("event quantiles must be positive integers")
    if "event_id" not in frame:
        frame["event_id"] = np.arange(len(frame))
    if frame["event_id"].isna().any() or frame["event_id"].duplicated().any():
        raise ValueError("event_id must be unique and nonmissing")
    return frame


def _value_frame(
    values: pl.DataFrame | pd.DataFrame, date_col: str, asset_col: str, value_col: str
) -> pd.DataFrame:
    if isinstance(values, pl.DataFrame):
        _validate_observation_keys(values, [date_col], "event values", unique=False)
        data = values.to_pandas()
        if asset_col in data:
            _validate_observation_keys(values, [date_col, asset_col], "event values")
            data = data.pivot(index=date_col, columns=asset_col, values=value_col)
        else:
            data = data.set_index(date_col)
    elif isinstance(values, pd.DataFrame):
        data = values.copy(deep=True)
    else:
        raise TypeError("values must be a wide dated Pandas frame or Polars value frame")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("event values require a DatetimeIndex")
    if (
        data.empty
        or data.index.hasnans
        or not data.index.is_unique
        or not data.index.is_monotonic_increasing
    ):
        raise ValueError("event value clock must be nonempty, increasing, unique and nonmissing")
    if not data.columns.is_unique:
        raise ValueError("event value asset columns must be unique")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in data.dtypes):
        raise TypeError("event values must be numeric")
    data.index = data.index.as_unit("ns")
    if np.isinf(data.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("event values must not contain infinity")
    return data


def align_event_paths(
    events: pl.DataFrame | pd.DataFrame | pd.Series,
    values: pl.DataFrame | pd.DataFrame,
    *,
    before: int = 10,
    after: int = 15,
    input_kind: Literal["levels", "returns"],
    demean_by: pl.DataFrame | pd.DataFrame | pd.Series | None = None,
    date_col: str = "date",
    asset_col: str = "asset",
    value_col: str = "value",
) -> pl.DataFrame:
    """Align each event to observed-clock offsets, retaining missing cells.

    ``levels`` passes supplied prices/wealth through without event-zero rebasing,
    matching legacy ``cumulative=True``. ``returns`` compounds the complete
    supplied clock from one with the existing NaN-as-flat wealth convention,
    matching legacy ``cumulative=False``. Demeaning uses the universe declared
    at each event timestamp, held fixed across that event's window.
    """
    for name, count in [("before", before), ("after", after)]:
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if input_kind not in ("levels", "returns"):
        raise ValueError("input_kind must be levels or returns")
    frame = _event_frame(events, date_col, asset_col)
    data = _value_frame(values, date_col, asset_col, value_col)
    if (frame["event_date"].dt.tz is None) != (data.index.tz is None):
        raise ValueError("event and value clocks must have matching timezone awareness")
    if data.index.tz is not None:
        frame["event_date"] = frame["event_date"].dt.tz_convert(data.index.tz)
    universe = None if demean_by is None else _event_frame(demean_by, date_col, asset_col)
    if universe is not None:
        if (universe["event_date"].dt.tz is None) != (data.index.tz is None):
            raise ValueError("demeaning and value clocks must have matching timezone awareness")
        if data.index.tz is not None:
            universe["event_date"] = universe["event_date"].dt.tz_convert(data.index.tz)
    for selected in [frame, *([] if universe is None else [universe])]:
        if not selected["asset"].isin(data.columns).all():
            raise ValueError("event universe contains assets absent from values")
    if input_kind == "returns":
        data = data.apply(
            lambda column: _safe_cumprod(1 + column.to_numpy(dtype=float, na_value=np.nan))
        )
    outcomes = data.to_numpy(dtype=float, na_value=np.nan)
    asset_indices = {asset: index for index, asset in enumerate(data.columns)}
    universes = (
        {}
        if universe is None
        else {
            timestamp: [asset_indices[asset] for asset in part["asset"]]
            for timestamp, part in universe.groupby("event_date", sort=False)
        }
    )
    benchmarks = {}
    rows = []
    for event in frame.to_dict("records"):
        timestamp = event["event_date"]
        position = data.index.get_indexer([timestamp])[0]
        peers = None
        if universe is not None and position >= 0:
            peers = universes.get(timestamp, [])
            if not peers:
                raise ValueError("demeaning universe is missing an event timestamp")
        for offset in range(-before, after + 1):
            index = position + offset
            source_date = raw = adjusted = benchmark = None
            status = "event_not_on_clock" if position < 0 else "window_clipped"
            if position >= 0 and 0 <= index < len(data):
                source_date = data.index[index]
                raw = float(outcomes[index, asset_indices[event["asset"]]])
                if peers is None:
                    benchmark = 0.0
                else:
                    key = (timestamp, offset)
                    if key not in benchmarks:
                        sample = outcomes[index, peers]
                        benchmarks[key] = _sample_return_statistics(sample[~np.isnan(sample)])[0]
                    benchmark = benchmarks[key]
                adjusted = raw - benchmark
                status = "observed" if np.isfinite(adjusted) else "missing_value"
                if not np.isfinite(adjusted):
                    adjusted = None
            rows.append(
                {
                    **{
                        key: event[key]
                        for key in [
                            "event_id",
                            "event_date",
                            "asset",
                            "quantile",
                            *(["group"] if "group" in frame else []),
                        ]
                    },
                    "offset": offset,
                    "source_date": source_date,
                    "raw_value": raw,
                    "benchmark": benchmark,
                    "value": adjusted,
                    "status": status,
                }
            )
    output = pd.DataFrame(rows)
    output["source_date"] = pd.DatetimeIndex(output["source_date"], tz=data.index.tz).as_unit("ns")
    return pl.from_pandas(output).with_columns(
        pl.col("raw_value", "benchmark", "value").cast(pl.Float64)
    )


def analyze_signal_events(
    events: pl.DataFrame | pd.DataFrame | pd.Series,
    values: pl.DataFrame | pd.DataFrame,
    *,
    before: int = 10,
    after: int = 15,
    input_kind: Literal["levels", "returns"],
    demeaned: bool = True,
    group_adjust: bool = False,
    by_group: bool = False,
    mean_by_date: bool = True,
    date_col: str = "date",
    asset_col: str = "asset",
    value_col: str = "value",
) -> AlignedEventResult:
    """Analyze supplied-value event paths with explicit asset/date/group weights.

    Group-adjusted date means are separate observations for each event-date/group,
    even when the final summary combines groups. This matches the legacy average
    path weighting order. With mean_by_date=False, individual events are samples.
    Duplicate event rows retain their unique IDs rather than being deduplicated.
    """
    for flag in [demeaned, group_adjust, by_group, mean_by_date]:
        if not isinstance(flag, bool):
            raise TypeError("event analysis options must be bool")
    frame = _event_frame(events, date_col, asset_col)
    if (group_adjust or by_group) and ("group" not in frame or frame["group"].isna().any()):
        raise ValueError("grouped event analysis requires complete group labels")
    normalized = frame.rename(columns={"event_date": "date"})
    if isinstance(values, pl.DataFrame):
        values = values.rename(
            {
                key: replacement
                for key, replacement in [(date_col, "date"), (asset_col, "asset")]
                if key in values.columns and key != replacement
            }
        )
    parts = (
        [group for _, group in normalized.groupby("group", observed=True, sort=False)]
        if group_adjust
        else [normalized]
    )
    paths = pl.concat(
        [
            align_event_paths(
                part,
                values,
                before=before,
                after=after,
                input_kind=input_kind,
                demean_by=part if (group_adjust or demeaned) else None,
                date_col="date",
                asset_col="asset",
                value_col=value_col,
            )
            for part in parts
        ],
        how="vertical_relaxed",
    ).sort("event_id", "offset")
    keys = ["quantile", *(["group"] if by_group else []), "offset"]
    sample_keys = list(dict.fromkeys([*keys, "event_date", *(["group"] if group_adjust else [])]))
    if mean_by_date:
        observed = []
        for part in paths.partition_by(sample_keys, maintain_order=True):
            valid = part["value"].drop_nulls().to_numpy()
            mean, _, count, _ = _sample_return_statistics(valid)
            observed.append(
                part.head(1)
                .select(sample_keys)
                .with_columns(
                    pl.lit(mean if count else None, dtype=pl.Float64).alias("value"),
                    pl.lit(count).alias("n_assets"),
                )
            )
        observations = pl.concat(observed)
    else:
        observations = paths.select("event_id", *sample_keys, "value").with_columns(
            pl.col("value").is_not_null().cast(pl.Int64).alias("n_assets")
        )
    statistics = []
    for part in observations.partition_by(keys, maintain_order=True):
        valid = part["value"].drop_nulls().to_numpy()
        mean, std, count, _ = _sample_return_statistics(valid)
        statistics.append(
            part.head(1)
            .select(keys)
            .with_columns(
                pl.lit(mean if count else None, dtype=pl.Float64).alias("mean"),
                pl.lit(std if count > 1 else None, dtype=pl.Float64).alias("std"),
                pl.lit(count).alias("n_obs"),
                pl.lit(part.height).alias("n_requested"),
            )
        )
    count_keys = ["date", "quantile", *(["group"] if "group" in normalized else [])]
    counts = pl.from_pandas(normalized).group_by(count_keys).len(name="n_events").sort(count_keys)
    tables = [paths, observations, pl.concat(statistics).sort(keys), counts]
    metadata = {
        "input_kind": input_kind,
        "units": "supplied_levels" if input_kind == "levels" else "wealth_starting_at_one",
        "event_zero_rebased": False,
        "window_unit": "observed_rows",
        "calendar_validation": "not performed",
        "before": int(before),
        "after": int(after),
        "demeaned": demeaned,
        "group_adjust": group_adjust,
        "by_group": by_group,
        "mean_by_date": mean_by_date,
        "sample_unit": "group_event_date"
        if mean_by_date and group_adjust
        else "event_date"
        if mean_by_date
        else "event",
        "std_definition": "sample standard deviation, ddof=1; not SE or CI",
        "duplicate_event_policy": "retain each event_id; no implicit deduplication",
        "missing_return_policy": "NaN-as-flat during full-clock compounding; no imputation of supplied levels",
    }
    return AlignedEventResult(
        tuple(table.write_ipc(None).getvalue() for table in tables),
        json.dumps(metadata, allow_nan=False),
    )
