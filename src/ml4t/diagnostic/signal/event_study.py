"""Compose signal event studies from owned preparation and existing analytical kernels."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from io import BytesIO
from numbers import Integral
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import polars as pl

from ml4t.diagnostic.signal.events import EventReturnsResult, analyze_event_returns

if TYPE_CHECKING:
    from ml4t.diagnostic.signal.panel import PreparedSignalPanel

_TABLES = (
    "factor_statistics",
    "event_distribution",
    "mean_returns",
    "return_observations",
    "prepared_events",
    "event_membership",
)


@dataclass(frozen=True)
class SignalEventStudyResult:
    """Owned factor stats, elapsed event bins and actual dated return samples."""

    _tables: tuple[bytes, ...] = field(repr=False)
    _metadata: str = field(repr=False)
    event_returns: EventReturnsResult | None = None

    @property
    def factor_statistics(self) -> pl.DataFrame:
        """Return factor-value statistics and counts for each supplied bucket."""
        return pl.read_ipc(BytesIO(self._tables[0]))

    @property
    def event_distribution(self) -> pl.DataFrame:
        """Return elapsed-time bin labels and declared-event counts, including empty bins."""
        return pl.read_ipc(BytesIO(self._tables[1]))

    @property
    def mean_returns(self) -> pl.DataFrame:
        """Return equal-date raw statistics plus separately labelled display means."""
        return pl.read_ipc(BytesIO(self._tables[2]))

    @property
    def return_observations(self) -> pl.DataFrame:
        """Return actual date/bucket means, counts and optional rate conversion."""
        return pl.read_ipc(BytesIO(self._tables[3]))

    @property
    def prepared_events(self) -> pl.DataFrame:
        """Return the owned complete prepared input, not only aggregate statistics."""
        return pl.read_ipc(BytesIO(self._tables[4]))

    @property
    def event_membership(self) -> pl.DataFrame:
        """Return each actual event key's elapsed-time bin without losing input order."""
        return pl.read_ipc(BytesIO(self._tables[5]))

    @property
    def metadata(self) -> dict:
        """Return independent horizon, conversion, elapsed-bin and row-window declarations."""
        return json.loads(self._metadata)

    def to_json(self) -> str:
        """Export all typed tables and optional paths without rounding or calculation."""
        return json.dumps(
            {
                "format_version": 1,
                "tables": {
                    name: base64.b64encode(table).decode("ascii")
                    for name, table in zip(_TABLES, self._tables, strict=True)
                },
                "metadata": self.metadata,
                "event_returns": None
                if self.event_returns is None
                else self.event_returns.to_json(),
            },
            allow_nan=False,
        )

    @classmethod
    def from_json(cls, value: str) -> SignalEventStudyResult:
        """Restore owned tables and optional aligned paths with timestamp precision."""
        data = json.loads(value)
        if data.get("format_version") != 1 or set(data.get("tables", {})) != set(_TABLES):
            raise ValueError("unsupported signal event-study result format")
        tables = tuple(base64.b64decode(data["tables"][name], validate=True) for name in _TABLES)
        for table in tables:
            pl.read_ipc(BytesIO(table))
        paths = data["event_returns"]
        return cls(
            tables,
            json.dumps(data["metadata"], allow_nan=False),
            None if paths is None else EventReturnsResult.from_json(paths),
        )


def _display_returns(frame: pl.DataFrame, ratios: dict[str, float]) -> pl.DataFrame:
    displayed = []
    for row in frame.select("return_column", "mean").iter_rows(named=True):
        value, ratio = row["mean"], ratios[row["return_column"]]
        if value is None or ratio == 1:
            converted = value
        else:
            with np.errstate(invalid="ignore", over="ignore"):
                converted = (
                    float(np.expm1(np.log1p(value) * ratio))
                    if value > -1
                    else float(np.power(1 + value, ratio) - 1)
                )
        displayed.append(converted if converted is not None and np.isfinite(converted) else None)
    return frame.with_columns(
        pl.Series("display_mean", displayed, dtype=pl.Float64),
        pl.col("return_column").replace_strict(ratios).alias("conversion_ratio"),
    ).with_columns(
        pl.when(pl.col("mean").is_null())
        .then(pl.lit("missing_mean"))
        .when(pl.col("display_mean").is_null())
        .then(pl.lit("undefined_rate"))
        .otherwise(pl.lit("defined"))
        .alias("display_status")
    )


def analyze_signal_event_study(
    panel: PreparedSignalPanel,
    values: pl.DataFrame | pd.DataFrame | None = None,
    *,
    input_kind: Literal["levels", "returns"] | None = None,
    event_window: tuple[int, int] | None = (5, 15),
    rate_of_return: bool = True,
    horizon_durations: Mapping[str, str | pd.Timedelta] | None = None,
    num_bars: int = 50,
) -> SignalEventStudyResult:
    """Compose one event-study sample without inferring calendars or re-bucketing.

    Event bins use elapsed time. Optional path windows count observed rows.
    Return panels use unadjusted, equal-date quantile statistics. Rate conversion
    requires explicit positive durations for multiple horizons and uses the first
    supplied horizon as its base. Raw statistics remain in their original units.
    """
    from ml4t.diagnostic.signal.panel import PreparedSignalPanel

    if not isinstance(panel, PreparedSignalPanel):
        raise TypeError("panel must be an owned PreparedSignalPanel")
    data = panel.data
    if data.is_empty():
        raise ValueError("event study requires at least one prepared event")
    if not isinstance(rate_of_return, bool):
        raise TypeError("rate_of_return must be bool")
    if isinstance(num_bars, bool) or not isinstance(num_bars, Integral) or num_bars < 1:
        raise ValueError("num_bars must be a positive integer")
    columns = panel.metadata["return_columns"]
    ratios = dict.fromkeys(columns, 1.0)
    durations = {}
    if horizon_durations is not None:
        if not isinstance(horizon_durations, Mapping):
            raise TypeError("horizon_durations must be a mapping")
        for column in columns:
            duration = horizon_durations.get(column)
            if not isinstance(duration, str | pd.Timedelta):
                raise ValueError(
                    "horizon_durations must declare positive explicit durations for every return column"
                )
            parsed = pd.Timedelta(duration)
            if pd.isna(parsed) or parsed <= pd.Timedelta(0):
                raise ValueError("horizon_durations must be positive and finite")
            durations[column] = int(parsed.value)
    if rate_of_return and len(columns) > 1 and not durations:
        raise ValueError("horizon_durations is required for multi-horizon rate conversion")
    if rate_of_return and durations:
        ratios = {column: durations[columns[0]] / durations[column] for column in columns}
    means = _display_returns(panel.quantile_statistics(demeaned=False), ratios)
    observations = _display_returns(panel.quantile_statistics(by_date=True, demeaned=False), ratios)
    factors = (
        data.group_by("quantile")
        .agg(
            pl.col(panel.factor_col).min().alias("min"),
            pl.col(panel.factor_col).max().alias("max"),
            pl.col(panel.factor_col).mean().alias("mean"),
            pl.col(panel.factor_col).std(ddof=1).alias("std"),
            pl.len().alias("count"),
        )
        .with_columns((pl.col("count") / pl.col("count").sum() * 100).alias("count_percent"))
        .sort("quantile")
    )
    dates = pd.DatetimeIndex(data[panel.date_col].to_pandas()).as_unit("ns")
    events = pd.DataFrame(
        {"event_row": np.arange(len(dates)), "factor": data[panel.factor_col].to_numpy()},
        index=dates,
    )
    span = dates.max() - dates.min()
    width = span / int(num_bars)
    bin_labels: list[pd.Timestamp | None]
    if span == pd.Timedelta(0):
        counts = pd.Series([len(events)], index=pd.DatetimeIndex([dates[0]]))
        bin_labels = [dates[0]] * len(events)
    else:
        if width <= pd.Timedelta(0):
            raise ValueError("num_bars exceeds the timestamp resolution of the event span")
        grouped = events.groupby(pd.Grouper(freq=width))
        counts = grouped["factor"].count()
        bin_labels = [None] * len(events)
        for label, part in grouped:
            for position in part["event_row"]:
                bin_labels[position] = label
    bin_starts = pl.from_pandas(
        pd.Series(pd.DatetimeIndex(bin_labels).as_unit("ns"), name="bin_start")
    )
    membership = data.select(
        pl.col(panel.date_col).alias("date"), pl.col(panel.asset_col).alias("asset"), "quantile"
    ).with_columns(bin_starts)
    distribution = pl.from_pandas(
        pd.DataFrame({"date": counts.index, "n_events": counts.to_numpy()})
    )
    paths = None
    if values is not None and event_window is not None:
        if input_kind not in ("levels", "returns"):
            raise ValueError(
                "input_kind must explicitly identify levels or returns for event paths"
            )
        before, after = event_window
        paths = analyze_event_returns(
            data,
            values,
            before=before,
            after=after,
            input_kind=input_kind,
            demeaned=False,
            date_col=panel.date_col,
            asset_col=panel.asset_col,
        )
    metadata = {
        "return_columns": columns,
        "input_columns": {
            "factor": panel.factor_col,
            "date": panel.date_col,
            "asset": panel.asset_col,
        },
        "rate_of_return": rate_of_return,
        "base_period": columns[0],
        "horizon_durations_ns": durations,
        "conversion_ratios": ratios,
        "raw_statistics_units": "supplied horizon returns",
        "display_units": "base-horizon rate" if rate_of_return else "supplied horizon returns",
        "sample_unit": "date",
        "return_adjustment": "none",
        "distribution_unit": "elapsed",
        "requested_num_bars": int(num_bars),
        "bin_width_ns": int(width.value),
        "singleton_bin_policy": "one bin at the supplied timestamp",
        "event_window_unit": "observed_rows",
        "calendar_validation": "not performed",
        "event_paths_available": paths is not None,
        "event_paths_availability": "provided"
        if paths is not None
        else "values_not_supplied"
        if values is None
        else "window_disabled",
        "base_rate_precision": "identity ratios retain the original mean; other positive-wealth rates use log1p/expm1",
    }
    tables = (factors, distribution, means, observations, data, membership)
    return SignalEventStudyResult(
        tuple(frame.write_ipc(None).getvalue() for frame in tables),
        json.dumps(metadata, allow_nan=False),
        paths,
    )
