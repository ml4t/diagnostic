"""Owned information-report samples from the existing date/group IC calculations."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

from ml4t.diagnostic.metrics.ic_inference import compute_ic_summary_stats

if TYPE_CHECKING:
    from ml4t.diagnostic.results.signal_results import SignalICResult
    from ml4t.diagnostic.signal.panel import PreparedSignalPanel

_TABLES = (
    "prepared_data",
    "ic_by_date",
    "group_ic_by_date",
    "summary",
    "monthly_ic",
    "mean_group_ic",
)


@dataclass(frozen=True)
class SignalInformationResult:
    """Independent typed samples, inference and aggregates for one information report."""

    _tables: tuple[bytes, ...] = field(repr=False)
    _metadata: str = field(repr=False)

    def get_table(self, name: str) -> pl.DataFrame:
        """Read an independent copy of one named analytical table."""
        if name not in _TABLES:
            raise ValueError(f"Unknown information table {name!r}; choose from {_TABLES}")
        return pl.read_ipc(BytesIO(self._tables[_TABLES.index(name)]))

    @property
    def metadata(self) -> dict:
        """Read independent selectors, grouping, sample and inference declarations."""
        return json.loads(self._metadata)

    def to_json(self) -> str:
        """Export all original typed samples and calculated tables without recomputing."""
        return json.dumps(
            {
                "format_version": 1,
                "tables": {
                    name: base64.b64encode(value).decode("ascii")
                    for name, value in zip(_TABLES, self._tables, strict=True)
                },
                "metadata": self.metadata,
            },
            allow_nan=False,
        )

    @classmethod
    def from_json(cls, value: str) -> SignalInformationResult:
        """Restore owned tables, preserving nanosecond dates and group labels."""
        data = json.loads(value)
        if data.get("format_version") != 1 or set(data.get("tables", {})) != set(_TABLES):
            raise ValueError("unsupported signal information result format")
        tables = tuple(base64.b64decode(data["tables"][name], validate=True) for name in _TABLES)
        for table in tables:
            pl.read_ipc(BytesIO(table))
        return cls(tables, json.dumps(data["metadata"], allow_nan=False))

    def to_ic_result(self) -> SignalICResult:
        """Adapt the global dated sample to existing charts, keeping undefined-date gaps."""
        from ml4t.diagnostic.results.signal_results import SignalICResult

        data, summary = self.get_table("ic_by_date"), self.get_table("summary")
        columns = self.metadata["return_columns"]
        first = data.filter(pl.col("return_column") == columns[0])
        values = {
            column: data.filter(pl.col("return_column") == column)["ic"].to_numpy().tolist()
            for column in columns
        }
        fields = {
            "ic_mean": "mean_ic",
            "ic_std": "std_ic",
            "ic_t_stat": "t_stat",
            "ic_p_value": "p_value",
            "ic_ir": "risk_adjusted_ic",
        }
        converted = {
            name: dict(zip(summary["return_column"], summary[column], strict=True))
            for name, column in fields.items()
        }
        return SignalICResult(
            ic_by_date=values,
            dates=[pd.Timestamp(value).isoformat() for value in first["date"]],
            ic_positive_pct={
                row["return_column"]: float("nan")
                if row["pct_positive"] is None
                else 100 * row["pct_positive"]
                for row in summary.iter_rows(named=True)
            },
            **converted,
        )


def _aggregate_ic(data: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    return (
        data.group_by(keys, maintain_order=True)
        .agg(
            pl.col("ic").mean().alias("mean_ic"),
            pl.len().alias("n_dates"),
            pl.col("ic").count().alias("n_defined"),
        )
        .with_columns(
            pl.when(pl.col("n_defined") == 0)
            .then(pl.lit("no_defined_ic"))
            .otherwise(pl.lit("defined"))
            .alias("status")
        )
    )


def analyze_signal_information(
    panel: PreparedSignalPanel,
    *,
    group_adjust: bool = False,
    by_group: bool = False,
    method: str = "spearman",
    min_obs: int = 2,
) -> SignalInformationResult:
    """Compose global and optional group ICs, naive inference and civil-month means.

    Group adjustment demeans outcomes before correlations. Report grouping adds
    group ICs without replacing the global sample. Counts and undefined outcomes
    come from the existing IC kernel. Supplied horizon labels do not declare time
    units, calendars or annualization. Monthly grouping uses supplied civil dates.
    """
    from ml4t.diagnostic.signal.panel import PreparedSignalPanel

    if not isinstance(panel, PreparedSignalPanel):
        raise TypeError("panel must be an owned PreparedSignalPanel")
    if not isinstance(by_group, bool):
        raise ValueError("by_group must be boolean")
    data = panel.data
    if data.is_empty():
        raise ValueError("information analysis requires at least one prepared observation")
    ic = panel.ic_by_date(group_adjust=group_adjust, method=method, min_obs=min_obs)
    groups = (
        panel.ic_by_date(group_adjust=group_adjust, by_group=True, method=method, min_obs=min_obs)
        if by_group
        else pl.DataFrame(schema={**ic.schema, "group": pl.String})
    )
    rows = []
    for part in ic.partition_by("return_column", maintain_order=True):
        clean = part["ic"].drop_nulls().to_numpy()
        summary = compute_ic_summary_stats(clean)
        deviation = summary["std_ic"]
        count = len(clean)
        varied = count > 1 and deviation > 0
        values = {
            "return_column": part["return_column"][0],
            "n_dates": part.height,
            "n_defined": count,
            "mean_ic": summary["mean_ic"],
            "std_ic": deviation,
            "risk_adjusted_ic": summary["mean_ic"] / deviation if varied else None,
            "t_stat": summary["t_stat"],
            "p_value": summary["p_value"],
            "pct_positive": summary["pct_positive"],
            "skew": float(stats.skew(clean)) if varied else None,
            "kurtosis": float(stats.kurtosis(clean)) if varied else None,
            "status": "no_defined_ic"
            if count == 0
            else "point_only"
            if count == 1
            else "constant_ic"
            if not varied
            else "defined",
        }
        rows.append(
            {
                name: None if isinstance(value, float) and not np.isfinite(value) else value
                for name, value in values.items()
            }
        )
    float_columns = [
        "mean_ic",
        "std_ic",
        "risk_adjusted_ic",
        "t_stat",
        "p_value",
        "pct_positive",
        "skew",
        "kurtosis",
    ]
    summary_table = pl.DataFrame(rows, schema_overrides=dict.fromkeys(float_columns, pl.Float64))
    monthly = _aggregate_ic(
        ic.with_columns(
            pl.col("date").dt.year().alias("year"), pl.col("date").dt.month().alias("month")
        ),
        ["return_column", "year", "month"],
    )
    mean_group = _aggregate_ic(groups, ["return_column", "group"])
    metadata = {
        "return_columns": panel.metadata["return_columns"],
        "group_adjust": group_adjust,
        "by_group": by_group,
        "method": method,
        "min_obs": int(min_obs),
        "sample_unit": "date",
        "monthly_unit": "supplied civil year/month",
        "inference": "naive two-sided one-sample t; sample standard deviation; no HAC",
        "degenerate_inference": "undefined for fewer than two or zero-variance ICs",
        "skew_kurtosis": "biased central moments; Fisher excess kurtosis",
        "calendar_validation": "not performed; no calendar or horizon inferred",
    }
    tables = [data, ic, groups, summary_table, monthly, mean_group]
    return SignalInformationResult(
        tuple(frame.write_ipc(None).getvalue() for frame in tables),
        json.dumps(metadata, allow_nan=False),
    )
