"""Dated portfolio regime samples over the owned account and native summaries."""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from io import BytesIO
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl

if TYPE_CHECKING:
    from .analysis import PortfolioAnalysis

_TABLES = ("samples", "statistics", "coverage", "transactions", "positions", "valuations")

# Historical names/date definitions used by Pyfolio; an open-ended event ends
# at the supplied sample, not a nondeterministic import-time system clock.
_PERIOD_DATES = {
    "Dotcom": ("2000-03-10", "2000-09-10"),
    "Lehman": ("2008-08-01", "2008-10-01"),
    "9/11": ("2001-09-11", "2001-10-11"),
    "US downgrade/European Debt Crisis": ("2011-08-05", "2011-09-05"),
    "Fukushima": ("2011-03-16", "2011-04-16"),
    "US Housing": ("2003-01-08", "2003-02-08"),
    "EZB IR Event": ("2012-09-10", "2012-10-10"),
    "Aug07": ("2007-08-01", "2007-09-01"),
    "Mar08": ("2008-03-01", "2008-04-01"),
    "Sept08": ("2008-09-01", "2008-10-01"),
    "2009Q1": ("2009-01-01", "2009-03-01"),
    "2009Q2": ("2009-03-01", "2009-06-01"),
    "Flash Crash": ("2010-05-05", "2010-05-10"),
    "Apr14": ("2014-04-01", "2014-05-01"),
    "Oct14": ("2014-10-01", "2014-11-01"),
    "Fall2015": ("2015-08-15", "2015-09-30"),
    "Low Volatility Bull Market": ("2005-01-01", "2007-08-01"),
    "GFC Crash": ("2007-08-01", "2009-04-01"),
    "Recovery": ("2009-04-01", "2013-01-01"),
    "New Normal": ("2013-01-01", "2018-09-21"),
    "Covid": ("2020-02-11", None),
}


@dataclass(frozen=True)
class PortfolioRegimeResult:
    """Owned actual regime membership, native summaries and availability.

    Stress windows may overlap deliberately; each membership retains its period.
    A live split instead partitions each supplied row once. Absent account
    components remain unavailable, not a known empty execution/position ledger.
    """

    _tables: tuple[bytes, ...] = field(repr=False)
    _metadata: str = field(repr=False)

    def _table(self, name: str) -> pl.DataFrame:
        return pl.read_ipc(BytesIO(self._tables[_TABLES.index(name)]))

    @property
    def samples(self) -> pl.DataFrame:
        """Actual dated returns, optional benchmark and requested regime membership."""
        return self._table("samples")

    @property
    def statistics(self) -> pl.DataFrame:
        """Existing native summary definitions at the declared common scale."""
        return self._table("statistics")

    @property
    def coverage(self) -> pl.DataFrame:
        """Every requested period, including unobserved or partly covered windows."""
        return self._table("coverage")

    @property
    def transactions(self) -> pl.DataFrame:
        """Actual executions partitioned without rounding or manufacturing event times."""
        return self._table("transactions")

    @property
    def positions(self) -> pl.DataFrame:
        """Supplied position rows only, without inferring complete snapshots."""
        return self._table("positions")

    @property
    def valuations(self) -> pl.DataFrame:
        """Supplied NAV/cash observations only; returns are not reconstructed."""
        return self._table("valuations")

    @property
    def metadata(self) -> dict[str, Any]:
        """Independent copy of sample, boundary, scale and account declarations."""
        return json.loads(self._metadata)

    def to_json(self) -> str:
        """Serialize exact Arrow types/instants and JSON method declarations."""
        return json.dumps(
            {
                "format": "ml4t.diagnostic.portfolio_regimes.v1",
                "metadata": self.metadata,
                "tables": {
                    name: base64.b64encode(value).decode("ascii")
                    for name, value in zip(_TABLES, self._tables, strict=True)
                },
            },
            allow_nan=False,
        )

    @classmethod
    def from_json(cls, value: str) -> PortfolioRegimeResult:
        """Restore the declared owned result without recalculating or filling data."""
        parsed = json.loads(value)
        if parsed["format"] != "ml4t.diagnostic.portfolio_regimes.v1":
            raise ValueError("unsupported portfolio regime result format")
        tables = tuple(base64.b64decode(parsed["tables"][name], validate=True) for name in _TABLES)
        for data in tables:
            pl.read_ipc(BytesIO(data))
        return cls(tables, json.dumps(parsed["metadata"], allow_nan=False))


def _pack(tables: Mapping[str, pl.DataFrame], metadata: dict[str, Any]) -> PortfolioRegimeResult:
    return PortfolioRegimeResult(
        tuple(tables[name].write_ipc(None).getvalue() for name in _TABLES),
        json.dumps(metadata, allow_nan=False),
    )


def _boundary(
    value: date | datetime | str, index: pd.DatetimeIndex, *, end_of_period: bool = False
) -> pd.Timestamp:
    civil = (
        isinstance(value, date)
        and not isinstance(value, datetime)
        or isinstance(value, str)
        and re.fullmatch(r"\d{4}(?:-\d{2})?(?:-\d{2})?", value) is not None
    )
    if civil and end_of_period:
        period = pd.Period(value)
        stamp = period.end_time
    else:
        stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("period boundaries must be nonmissing dates or instants")
    if civil and index.tz is not None:
        stamp = stamp.tz_localize(index.tz)
    elif (stamp.tzinfo is None) != (index.tz is None):
        raise ValueError(
            "instant boundaries and observations require compatible timezone declarations"
        )
    elif stamp.tzinfo is not None and index.tz is not None:
        stamp = stamp.tz_convert(index.tz)
    return stamp


def _window_mask(
    dates: pl.Series, start: Any, end: Any, *, end_exclusive: bool = False
) -> np.ndarray:
    index = pd.DatetimeIndex(dates.to_pandas())
    left = np.ones(len(index), dtype=bool) if start is None else index >= _boundary(start, index)
    if end is None:
        return np.asarray(left)
    stop = _boundary(end, index, end_of_period=not end_exclusive)
    right = index < stop if end_exclusive else index <= stop
    return np.asarray(left & right)


def _summaries(
    analysis: PortfolioAnalysis, samples: pl.DataFrame, label: str, names: list[str]
) -> pl.DataFrame:
    from .analysis import PortfolioAnalysis
    from .results import PortfolioMetrics

    records = []
    for name in names:
        frame = samples.filter(pl.col(label) == name)
        result = dict.fromkeys(PortfolioMetrics.__dataclass_fields__, None)
        if frame.height:
            subset = PortfolioAnalysis(
                frame["return"],
                benchmark=frame["benchmark"] if "benchmark" in frame.columns else None,
                dates=frame["date"],
                risk_free=analysis._risk_free,
                periods_per_year=analysis._periods_per_year,
            )
            result = subset.compute_summary_stats().to_dict()
        records.append({label: name, "n_observations": frame.height, **result})
    schema = {
        label: pl.String,
        "n_observations": pl.Int64,
        **dict.fromkeys(PortfolioMetrics.__dataclass_fields__, pl.Float64),
    }
    return pl.DataFrame(records, schema=schema)


def analyze_regimes(
    analysis: PortfolioAnalysis,
    *,
    periods: Mapping[str, tuple[date | datetime | str, date | datetime | str]] | None = None,
    live_start: date | datetime | str | None = None,
) -> PortfolioRegimeResult:
    """Select real dated samples once and reuse native portfolio summaries."""
    analysis.require_dates()
    if analysis.input_data is None and (analysis.has_positions or analysis.has_transactions):
        raise ValueError(
            "regime account components require owned PortfolioAnalysis.from_frames inputs"
        )
    dates = analysis.dates
    full = pl.DataFrame({"date": dates, "return": analysis.returns})
    if analysis.benchmark is not None:
        full = full.with_columns(pl.Series("benchmark", analysis.benchmark))
    if live_start is not None and periods is not None:
        raise ValueError("supply either stress periods or one live_start boundary")
    if live_start is not None:
        windows = {"pre_live": (None, live_start), "live": (live_start, None)}
        label = "regime"
    else:
        windows = dict(_PERIOD_DATES if periods is None else periods)
        label = "period"
    if any(not isinstance(name, str) or not name for name in windows):
        raise ValueError("periods must contain nonempty named windows")
    for name, bounds in windows.items():
        if not isinstance(bounds, tuple | list) or len(bounds) != 2:
            raise ValueError(f"period {name!r} requires one start/end pair")
        if bounds[0] is not None and bounds[1] is not None:
            index = pd.DatetimeIndex(dates.to_pandas())
            if _boundary(bounds[0], index) > _boundary(bounds[1], index, end_of_period=True):
                raise ValueError(f"period {name!r} ends before it starts")
    members, coverage = [], []
    for name, (start, end) in windows.items():
        selected = _window_mask(
            dates, start, end, end_exclusive=name == "pre_live" and live_start is not None
        )
        sample = full.filter(pl.Series(selected)).with_columns(pl.lit(name).alias(label))
        members.append(sample)
        index = pd.DatetimeIndex(dates.to_pandas())
        extends_left = start is not None and _boundary(start, index) < index[0]
        extends_right = (
            end is not None
            and _boundary(end, index, end_of_period=name != "pre_live" or live_start is None)
            > index[-1]
        )
        coverage.append(
            {
                label: name,
                "n_observations": sample.height,
                "status": "unobserved"
                if not sample.height
                else "partial"
                if extends_left or extends_right
                else "observed",
                "requested_start": str(start) if start is not None else None,
                "requested_end": str(end) if end is not None else None,
            }
        )
    samples = pl.concat(members) if members else full.head(0).with_columns(pl.lit("").alias(label))
    tables = {
        "samples": samples,
        "coverage": pl.DataFrame(
            coverage,
            schema={
                label: pl.String,
                "n_observations": pl.Int64,
                "status": pl.String,
                "requested_start": pl.String,
                "requested_end": pl.String,
            },
        ),
        "statistics": _summaries(analysis, samples, label, list(windows)),
    }
    inputs = analysis.input_data
    availability = {}
    for name in ("transactions", "positions", "valuations"):
        frame = getattr(inputs, name) if inputs is not None else None
        availability[name] = frame is not None
        if frame is None:
            tables[name] = pl.DataFrame(schema={label: pl.String})
            continue
        key = "timestamp" if name == "transactions" else "date"
        pieces = []
        for window, (start, end) in windows.items():
            mask = _window_mask(
                frame[key],
                start,
                end,
                end_exclusive=window == "pre_live" and live_start is not None,
            )
            pieces.append(frame.filter(pl.Series(mask)).with_columns(pl.lit(window).alias(label)))
        tables[name] = (
            pl.concat(pieces) if pieces else frame.head(0).with_columns(pl.lit("").alias(label))
        )
    metadata = {
        "kind": "live_split" if live_start is not None else "stress",
        "label_column": label,
        "periods_per_year": analysis._periods_per_year,
        "annual_risk_free": analysis._risk_free,
        "boundary": "pre_live < live_start; live >= live_start"
        if live_start is not None
        else "inclusive supplied endpoints; date endpoints include their civil day",
        "availability": availability,
        "account": inputs.metadata if inputs is not None else None,
        "default_periods": periods is None and live_start is None,
        "coverage_definition": "requested extent versus observed extent; no expected grid or missing-session inference",
        "open_event_end": "supplied observation extent, not import-time today",
        "returns": "owned finite native simple-return sample; no NAV reconstruction or calendar resampling",
    }
    return _pack(tables, metadata)
