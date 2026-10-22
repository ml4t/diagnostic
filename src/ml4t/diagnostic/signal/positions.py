"""Mathematical factor allocations on an explicit union of entry and expiry keys."""

from __future__ import annotations

import base64
import heapq
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import pandas as pd
import polars as pl
from ml4t.specs import UNSET_CALENDAR, CalendarUnset, FeedSpec

from ml4t.diagnostic.signal._prepared import signal_input_contract
from ml4t.diagnostic.signal._utils import _signal_frame
from ml4t.diagnostic.signal.portfolio import _select_factor_sample, compute_factor_weights
from ml4t.diagnostic.utils.validation import (
    _align_datetime_key_resolution,
    _validate_observation_keys,
)


@dataclass(frozen=True)
class SyntheticPositionsResult:
    """Owned allocations, union-event weights and explicit exposure states.

    Allocations are research weights, not orders or executed shares. Every
    selected date/asset observation has a stable allocation_id and supplied
    expiry. Positions include zero weights for the selected asset universe at
    each entry/expiry event. A timeline distinguishes no active allocation,
    zero signals and exact cancellation of nonzero allocations.
    """

    _allocations: bytes = field(repr=False)
    _positions: bytes = field(repr=False)
    _timeline: bytes = field(repr=False)
    _metadata: str = field(repr=False)

    @property
    def allocations(self) -> pl.DataFrame:
        """Return independent original weights, IDs and explicit expiry keys."""
        return pl.read_ipc(BytesIO(self._allocations))

    @property
    def positions(self) -> pl.DataFrame:
        """Return independent long date/asset/net-gross-normalized weights."""
        return pl.read_ipc(BytesIO(self._positions))

    @property
    def timeline(self) -> pl.DataFrame:
        """Return independent union-event counts and exposure states."""
        return pl.read_ipc(BytesIO(self._timeline))

    @property
    def metadata(self) -> dict:
        """Return independent input declarations and analytical definitions."""
        return json.loads(self._metadata)

    def to_json(self) -> str:
        """Serialize exact typed Arrow keys and strict JSON metadata."""
        return json.dumps(
            {
                "format_version": 1,
                "tables": {
                    name: base64.b64encode(getattr(self, f"_{name}")).decode("ascii")
                    for name in ("allocations", "positions", "timeline")
                },
                "metadata": self.metadata,
            },
            allow_nan=False,
        )

    @classmethod
    def from_json(cls, value: str) -> SyntheticPositionsResult:
        """Restore owned tables, rejecting unsupported result formats."""
        parsed = json.loads(value)
        if parsed.get("format_version") != 1:
            raise ValueError("unsupported synthetic positions format_version")
        tables = [
            base64.b64decode(parsed["tables"][name], validate=True)
            for name in ("allocations", "positions", "timeline")
        ]
        for payload in tables:
            pl.read_ipc(BytesIO(payload))
        return cls(tables[0], tables[1], tables[2], json.dumps(parsed["metadata"], allow_nan=False))


def _expiry_keys(weights: pl.DataFrame, expiries: pl.DataFrame, expiry_col: str) -> pl.DataFrame:
    _validate_observation_keys(expiries, ["date", "asset"], "allocation expiries")
    if expiry_col not in expiries.columns:
        raise ValueError(f"expiry column {expiry_col!r} was not found")
    weights, expiries = _align_datetime_key_resolution(weights, expiries, "date")
    if weights["date"].dtype != expiries["date"].dtype:
        raise ValueError("allocation entry and expiry lookup date keys must have matching types")
    weights = weights.join(
        expiries.select("date", "asset", pl.col(expiry_col).alias("expiry")),
        on=["date", "asset"],
        how="left",
        maintain_order="left",
    )
    dtype = weights["date"].dtype
    expiry_type = weights["expiry"].dtype
    if dtype != pl.Date and not isinstance(dtype, pl.Datetime):
        raise TypeError("allocation date keys must contain Date or Datetime values")
    # Align precision without converting civil session labels into instants.
    if isinstance(dtype, pl.Datetime) and isinstance(expiry_type, pl.Datetime):
        left, right = _align_datetime_key_resolution(
            weights.select("date"), weights.select(pl.col("expiry").alias("date")), "date"
        )
        weights = weights.with_columns(left["date"], right["date"].alias("expiry"))
    if weights["date"].dtype != weights["expiry"].dtype:
        raise ValueError("expiry must use the same civil/instant and timezone type as entry")
    if weights["expiry"].null_count() or (weights["expiry"] <= weights["date"]).any():
        raise ValueError("expiry must be known and strictly after each allocation entry")
    return weights.with_row_index("allocation_id")


def _aggregate_allocations(allocations: pl.DataFrame, metadata: dict) -> SyntheticPositionsResult:
    """Use integer keys so Python datetime conversion cannot truncate nanoseconds."""
    dtype = allocations["date"].dtype
    assets = allocations["asset"].unique(maintain_order=True)
    codes = {asset: code for code, asset in enumerate(assets)}
    entries: dict[int, list[tuple[int, int, int, float]]] = {}
    event_keys = set()
    for identifier, entry, asset, weight, expiry in allocations.select(
        "allocation_id",
        pl.col("date").cast(pl.Int64),
        "asset",
        "weight",
        pl.col("expiry").cast(pl.Int64),
    ).iter_rows():
        entries.setdefault(entry, []).append((identifier, expiry, codes[asset], weight))
        event_keys.update((entry, expiry))
    active: dict[int, tuple[int, float]] = {}
    expiry_queue: list[tuple[int, int]] = []
    positions, timeline = [], []
    for key in sorted(event_keys):
        while expiry_queue and expiry_queue[0][0] <= key:
            _, identifier = heapq.heappop(expiry_queue)
            del active[identifier]
        for identifier, expiry, code, weight in entries.get(key, []):
            active[identifier] = code, weight
            heapq.heappush(expiry_queue, (expiry, identifier))
        scale = max((abs(weight) for _, weight in active.values()), default=0.0)
        by_asset: list[list[float]] = [[] for _ in assets]
        if scale:
            for code, weight in active.values():
                by_asset[code].append(weight / scale)
        net = [math.fsum(values) for values in by_asset]
        gross = math.fsum(abs(weight) for weight in net)
        weights = [weight / gross if gross else 0.0 for weight in net]
        status = (
            "no_active_allocations"
            if not active
            else "zero_signal"
            if not scale
            else "offsetting"
            if not gross
            else "invested"
        )
        positions.extend(
            (key, asset, weight) for asset, weight in zip(assets, weights, strict=True)
        )
        timeline.append(
            (key, len(active), status, math.fsum(abs(w) for w in weights), math.fsum(weights))
        )
    position_frame = pl.DataFrame(
        positions,
        schema={"date": pl.Int64, "asset": assets.dtype, "weight": pl.Float64},
        orient="row",
    ).with_columns(pl.col("date").cast(dtype))
    timeline_frame = pl.DataFrame(
        timeline,
        schema={
            "date": pl.Int64,
            "active_allocations": pl.UInt64,
            "status": pl.String,
            "gross_exposure": pl.Float64,
            "net_exposure": pl.Float64,
        },
        orient="row",
    ).with_columns(pl.col("date").cast(dtype))
    return SyntheticPositionsResult(
        allocations.write_ipc(None).getvalue(),
        position_frame.write_ipc(None).getvalue(),
        timeline_frame.write_ipc(None).getvalue(),
        json.dumps(metadata, allow_nan=False),
    )


def compute_factor_positions(
    data: pl.DataFrame | pd.DataFrame,
    *,
    expiries: pl.DataFrame | pd.DataFrame | None = None,
    expiry_col: str = "expiry",
    demeaned: bool = True,
    group_adjust: bool = False,
    equal_weight: bool = False,
    quantiles: Sequence[int] | None = None,
    groups: Sequence[str] | None = None,
    factor_col: str = "factor",
    date_col: str | None = None,
    asset_col: str | None = None,
    group_col: str = "group",
    quantile_col: str = "quantile",
    feed_spec: FeedSpec | Mapping[str, Any] | None = None,
    timestamp_col: str | None = None,
    entity_col: str | None = None,
    calendar: str | CalendarUnset | None = UNSET_CALENDAR,
) -> SyntheticPositionsResult:
    """Select and size factors, then expire every allocation on supplied keys.

    Expiries must contain the same date/asset keys and a strictly later expiry
    of the same Date or Datetime/timezone type. With expiries=None, expiry_col
    is read from data. No selector string, return observation count or inferred
    calendar is used to manufacture an expiry. The declared calendar is retained
    as input metadata only; explicit None disables schedules. No orders, shares,
    prices, executions, transaction costs or mark-to-market returns are inferred.
    """
    date_col, asset_col, contract = signal_input_contract(
        feed_spec=feed_spec,
        timestamp_col=timestamp_col,
        entity_col=entity_col,
        date_col=date_col,
        asset_col=asset_col,
        calendar=calendar,
    )
    values = _signal_frame(data)
    selected = _select_factor_sample(values, quantiles, groups, quantile_col, group_col)
    weights = compute_factor_weights(
        selected,
        demeaned=demeaned,
        group_adjust=group_adjust,
        equal_weight=equal_weight,
        factor_col=factor_col,
        date_col=date_col,
        asset_col=asset_col,
        group_col=group_col,
    )
    expiry_values = _signal_frame(expiries) if expiries is not None else selected
    expiry_values = expiry_values.rename({date_col: "date", asset_col: "asset"})
    allocations = _expiry_keys(weights, expiry_values, expiry_col)
    return _aggregate_allocations(
        allocations,
        {
            "representation": "synthetic_expiring_allocations",
            "input_contract": contract,
            "expiry_source": expiry_col,
            "normalization": "active_net_unit_gross",
            "ordering": "expire_all_due_then_add_entries",
            "execution_account": False,
            "demeaned": bool(demeaned),
            "group_adjust": bool(group_adjust),
            "equal_weight": bool(equal_weight),
            "factor_col": factor_col,
            "group_col": group_col,
        },
    )
