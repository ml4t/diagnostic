"""Compose signal event studies from owned preparation and existing analytical kernels."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from io import BytesIO
from numbers import Integral, Real
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import polars as pl

from ml4t.diagnostic.signal.events import EventReturnsResult, analyze_event_returns
from ml4t.diagnostic.signal.quantile import _display_returns, _sample_return_statistics

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


def analyze_signal_event_study(
    panel: PreparedSignalPanel,
    values: pl.DataFrame | pd.DataFrame | None = None,
    *,
    input_kind: Literal["levels", "returns"] | None = None,
    event_window: tuple[int, int] | None = (5, 15),
    rate_of_return: bool = True,
    rate_conversion_factors: Mapping[str, float] | None = None,
    num_bars: int = 50,
) -> SignalEventStudyResult:
    """Compose one event-study sample without inferring calendars or re-bucketing.

    Event bins use elapsed time. Optional path windows count observed rows.
    Return panels use unadjusted, equal-date quantile statistics. Rate conversion
    requires explicit dimensionless powers for multiple horizons and uses the first
    supplied horizon as its base. Its factor must be one. For three-observation
    and one-observation returns, declare factors one and three. Column labels
    establish neither elapsed durations nor calendars. Raw statistics are retained.
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
    declared = {}
    if rate_conversion_factors is not None:
        if not isinstance(rate_conversion_factors, Mapping):
            raise TypeError("rate_conversion_factors must be a mapping")
        for column in columns:
            factor = rate_conversion_factors.get(column)
            if (
                isinstance(factor, bool)
                or not isinstance(factor, Real)
                or not np.isfinite(factor)
                or factor <= 0
            ):
                raise ValueError(
                    "rate_conversion_factors must declare a positive finite power for every return column"
                )
            declared[column] = float(factor)
        if declared[columns[0]] != 1:
            raise ValueError(
                "the first return column is the base horizon and must have conversion factor one"
            )
    if rate_of_return and len(columns) > 1 and not declared:
        raise ValueError("rate_conversion_factors is required for multi-horizon rate conversion")
    if rate_of_return and declared:
        ratios = declared
    means = _display_returns(panel.quantile_statistics(demeaned=False), ratios)
    observations = _display_returns(panel.quantile_statistics(by_date=True, demeaned=False), ratios)
    factor_rows = []
    for part in data.partition_by("quantile", maintain_order=True):
        values_in_bucket = part[panel.factor_col].to_numpy()
        mean, std, count, _ = _sample_return_statistics(values_in_bucket)
        factor_rows.append(
            {
                "quantile": part["quantile"][0],
                "min": float(np.min(values_in_bucket)),
                "max": float(np.max(values_in_bucket)),
                "mean": mean,
                "std": std if count > 1 else None,
                "count": count,
            }
        )
    factors = (
        pl.DataFrame(
            factor_rows,
            schema={
                "quantile": data.schema["quantile"],
                "min": pl.Float64,
                "max": pl.Float64,
                "mean": pl.Float64,
                "std": pl.Float64,
                "count": pl.UInt64,
            },
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
        "declared_conversion_factors": declared,
        "conversion_factor_unit": "dimensionless power; no time unit or calendar inferred",
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
