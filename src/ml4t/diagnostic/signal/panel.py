"""Owned common-horizon input preparation, separate from return construction."""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl

from ml4t.diagnostic.backends.polars_backend import PolarsBackend, _bucket_parameters
from ml4t.diagnostic.utils.validation import (
    _align_datetime_key_resolution,
    _validate_observation_keys,
)

if TYPE_CHECKING:
    from ml4t.diagnostic.signal.result import SignalResult


@dataclass(frozen=True)
class PreparedSignalPanel:
    """An owned prepared sample with independently accessible loss metadata."""

    _payload: bytes = field(repr=False)
    _audit: str = field(repr=False)
    factor_col: str
    date_col: str
    asset_col: str

    @property
    def data(self) -> pl.DataFrame:
        """Return a new frame without exposing the stored Arrow snapshot."""
        return pl.read_ipc(BytesIO(self._payload))

    @property
    def metadata(self) -> dict:
        """Return an independent copy of the preparation audit."""
        return json.loads(self._audit)

    @property
    def excluded_data(self) -> pl.DataFrame:
        """Return excluded keys with their first failing stage and reason.

        The Arrow snapshot preserves timestamp precision and timezone. Its
        base64 representation also survives the existing analysis JSON export.
        """
        return pl.read_ipc(BytesIO(base64.b64decode(self.metadata["excluded_data_ipc_base64"])))

    def analyze(
        self,
        *,
        min_assets: int = 10,
        ic_method: str = "spearman",
        compute_turnover_flag: bool = False,
        autocorrelation_lags: int = 10,
    ) -> SignalResult:
        """Feed supported integer-day selectors to the existing analysis engine.

        These labels select supplied columns; they do not establish durations or
        calendar conformance. Other horizon labels remain available in ``data``.
        """
        from ml4t.diagnostic.signal.core import analyze_prepared_signal

        columns = self.metadata["return_columns"]
        if not all(re.fullmatch(r"[1-9]\d*D", column) for column in columns):
            raise ValueError("analysis currently requires positive integer D horizon labels")
        data = self.data
        buckets = sorted(data["quantile"].unique().to_list())
        if not buckets or buckets != list(range(1, max(buckets) + 1)):
            raise ValueError("analysis requires a nonempty contiguous bucket range starting at1")
        result = analyze_prepared_signal(
            data,
            periods=tuple(int(column[:-1]) for column in columns),
            quantiles=max(buckets),
            min_assets=min_assets,
            ic_method=ic_method,
            compute_turnover_flag=compute_turnover_flag,
            autocorrelation_lags=autocorrelation_lags,
            factor_col=self.factor_col,
            date_col=self.date_col,
            asset_col=self.asset_col,
        )
        return replace(
            result, preparation_metadata={**result.preparation_metadata, "cleaning": self.metadata}
        )

    def ic_by_date(
        self,
        *,
        method: str = "spearman",
        group_adjust: bool = False,
        by_group: bool = False,
        min_obs: int = 2,
        date_grid: pl.DataFrame | pd.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Inspect ICs for this owned sample, with optional outcome/group adjustment.

        Unlike analyze(), arbitrary supplied return selectors are supported;
        they do not imply a calendar or duration. See compute_signal_ic for
        counts, undefined statuses and explicit requested-grid behavior.
        """
        from ml4t.diagnostic.signal.signal_ic import compute_signal_ic

        return compute_signal_ic(
            self.data,
            return_columns=self.metadata["return_columns"],
            method=method,
            group_adjust=group_adjust,
            by_group=by_group,
            min_obs=min_obs,
            factor_col=self.factor_col,
            date_col=self.date_col,
            asset_col=self.asset_col,
            date_grid=date_grid,
        )

    def factor_weights(
        self,
        *,
        demeaned: bool = True,
        group_adjust: bool = False,
        equal_weight: bool = False,
    ) -> pl.DataFrame:
        """Inspect factor exposure with explicit zero-signal handling.

        See compute_factor_weights for the single shared weight calculation.
        This constructs no dated holdings, capital or expiration accounting.
        """
        from ml4t.diagnostic.signal.portfolio import compute_factor_weights

        return compute_factor_weights(
            self.data,
            demeaned=demeaned,
            group_adjust=group_adjust,
            equal_weight=equal_weight,
            factor_col=self.factor_col,
            date_col=self.date_col,
            asset_col=self.asset_col,
        )

    def factor_returns(
        self,
        *,
        demeaned: bool = True,
        group_adjust: bool = False,
        equal_weight: bool = False,
        by_asset: bool = False,
        date_grid: pl.DataFrame | pd.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Inspect per-asset contributions or aggregate supplied-horizon returns.

        Weights do not depend on horizon-specific future availability. See
        compute_factor_returns for counts, missing tails and explicit output grids.
        """
        from ml4t.diagnostic.signal.portfolio import compute_factor_returns

        return compute_factor_returns(
            self.data,
            return_columns=self.metadata["return_columns"],
            demeaned=demeaned,
            group_adjust=group_adjust,
            equal_weight=equal_weight,
            by_asset=by_asset,
            date_grid=date_grid,
            factor_col=self.factor_col,
            date_col=self.date_col,
            asset_col=self.asset_col,
        )

    def quantile_statistics(
        self,
        *,
        by_date: bool = False,
        by_group: bool = False,
        demeaned: bool = True,
        group_adjust: bool = False,
    ) -> pl.DataFrame:
        """Inspect date/bucket sample statistics or equal-date aggregates.

        See compute_quantile_statistics for explicit asset/date counts, sample
        standard errors and group adjustment. The pooled analyze() defaults
        and its spread inference remain unchanged.
        """
        from ml4t.diagnostic.signal.quantile import compute_quantile_statistics

        return compute_quantile_statistics(
            self.data,
            return_columns=self.metadata["return_columns"],
            by_date=by_date,
            by_group=by_group,
            demeaned=demeaned,
            group_adjust=group_adjust,
            factor_col=self.factor_col,
            date_col=self.date_col,
            asset_col=self.asset_col,
        )

    def demean_returns(self, *, by_group: bool = False) -> pl.DataFrame:
        """Inspect date- or date/group-demeaned outcomes without changing this panel.

        Rows, keys, factors and buckets are retained. This uses the same paired
        outcome adjustment as group-adjusted IC, not portfolio-weight construction.
        """
        from ml4t.diagnostic.signal.returns import demean_signal_returns

        return demean_signal_returns(
            self.data,
            return_columns=self.metadata["return_columns"],
            by_group=by_group,
            factor_col=self.factor_col,
            date_col=self.date_col,
            asset_col=self.asset_col,
        )


def _frame(
    data: pl.DataFrame | pd.DataFrame | pd.Series,
    date_col: str,
    asset_col: str,
    value_col: str | None = None,
) -> pl.DataFrame:
    if isinstance(data, pd.Series):
        if value_col is None:
            raise TypeError("a return panel must be a DataFrame")
        data = data.rename(value_col).to_frame()
    if isinstance(data, pd.DataFrame):
        if isinstance(data.index, pd.MultiIndex):
            data = data.reset_index()
        frame = pl.from_pandas(data)
    elif isinstance(data, pl.DataFrame):
        frame = data.clone()
    else:
        raise TypeError("signal inputs must be Pandas or Polars frames, or indexed series")
    _validate_observation_keys(frame, [date_col, asset_col], "signal panel")
    if frame[date_col].dtype != pl.Date and not isinstance(frame[date_col].dtype, pl.Datetime):
        raise TypeError("signal panel dates must contain Date or Datetime values")
    return frame


def prepare_signal_panel(
    factor: pl.DataFrame | pd.DataFrame | pd.Series,
    forward_returns: pl.DataFrame | pd.DataFrame,
    *,
    quantiles: int | Sequence[float] | np.ndarray | None = 5,
    bins: int | Sequence[float] | np.ndarray | None = None,
    zero_aware: bool = False,
    return_columns: Sequence[str] | None = None,
    groups: Mapping | pl.DataFrame | pd.DataFrame | pd.Series | None = None,
    group_labels: Mapping | None = None,
    binning_by_group: bool = False,
    max_loss: float = 0.35,
    factor_col: str = "factor",
    date_col: str = "date",
    asset_col: str = "asset",
) -> PreparedSignalPanel:
    """Select one complete supplied-return sample, then assign qcut/cut buckets.

    Losses use the original factor count and disjoint factor, return, group and
    bucketing stages. Nonfinite factors/returns and missing group observations
    are excluded explicitly, not filled. A failed bucket group is excluded only
    when max_loss>0; max_loss=0 exposes its actual error. No returns or calendar
    are inferred. Existing price-based preparation defaults remain unchanged.

    Parameters
    ----------
    factor : DataFrame or pandas Series
        Numeric factor values with unique date/asset keys. Pandas MultiIndex
        names must match the supplied key names; flat frames are also accepted.
    forward_returns : DataFrame
        Precomputed numeric outcomes with the same key schema. No prices are
        needed. Missing/nonfinite requested outcomes are unavailable, not zeros.
    quantiles : int or sequence, optional
        Positive qcut count or increasing probabilities in [0,1]. Use None for
        cut bins. Partial probability ranges exclude observations outside them.
    bins : int or sequence, optional
        Positive equal-width count or increasing edges, mutually exclusive with
        quantiles. Unbounded endpoints use -inf/+inf strings in audit JSON.
    zero_aware : bool
        Bucket negative and nonnegative factors separately using floor(count/2)
        buckets on each side. Requires an integer count of at least two.
    return_columns : sequence of str, optional
        Columns defining the common sample; defaults to all non-key columns in
        forward_returns. Labels are selectors, not inferred calendar durations.
    groups : mapping, DataFrame or pandas Series, optional
        Static asset-to-group mapping or dated date/asset group values. Missing
        static asset mappings raise; missing dated groups contribute group losses.
    group_labels : mapping, optional
        Replace group codes before bucketing, as in Alphalens' groupby_labels.
    binning_by_group : bool
        Assign buckets within date/group rather than date alone. This does not
        alone provide group-neutral inference or factor portfolios. Request
        factor_weights(group_adjust=True) explicitly for group-normalized exposure.
    max_loss : float
        Maximum excluded fraction of original factor rows; equality passes.
        One permits an empty owned panel; analyze() still rejects empty samples.
        Zero exposes bucket errors and allows no exclusions.
    factor_col, date_col, asset_col : str
        Explicit input column/index names, distinct from group, quantile and
        the exclusion audit's exclusion_stage/exclusion_reason columns.

    Returns
    -------
    PreparedSignalPanel
        Owned Float64 factor/outcome values, exact retained/excluded keys and
        loss provenance. excluded_data exposes each exclusion's stage/reason.
        Float32 input quantization cannot be recovered. Original input hashes
        precede Float64 conversion; caller objects are not modified.
    """
    specification, bucket_count = _bucket_parameters(quantiles, bins, zero_aware)
    quantiles, bins = (specification, None) if bins is None else (None, specification)
    if not isinstance(binning_by_group, bool | np.bool_):
        raise ValueError("binning_by_group must be boolean")
    if (
        isinstance(max_loss, bool)
        or not isinstance(max_loss, Real)
        or not np.isfinite(max_loss)
        or not 0 <= max_loss <= 1
    ):
        raise ValueError("max_loss must be a finite fraction in[0,1]")
    if (
        len(
            {
                date_col,
                asset_col,
                factor_col,
                "group",
                "quantile",
                "exclusion_stage",
                "exclusion_reason",
            }
        )
        != 7
    ):
        raise ValueError(
            "signal key/value names must be distinct from group, quantile and exclusion columns"
        )
    factors = _frame(factor, date_col, asset_col, factor_col)
    returns = _frame(forward_returns, date_col, asset_col)
    keys = [date_col, asset_col]
    columns = (
        list(return_columns)
        if return_columns is not None
        else [column for column in returns.columns if column not in keys]
    )
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("return_columns must be nonempty and unique")
    if set(columns).intersection(
        [*keys, factor_col, f"{factor_col}_quantile", "group", "quantile"]
    ):
        raise ValueError("return columns must be distinct from signal keys and values")
    for frame, names in [(factors, [factor_col]), (returns, columns)]:
        for column in names:
            if column not in frame.columns or not frame[column].dtype.is_numeric():
                raise TypeError(f"signal panel {column} must contain numeric values")
    if factors.is_empty():
        raise ValueError("factor must contain observations")
    input_factor_hash = sha256(
        factors.select(*keys, factor_col).write_ipc(None).getvalue()
    ).hexdigest()
    input_returns_hash = sha256(
        returns.select(*keys, *columns).write_ipc(None).getvalue()
    ).hexdigest()
    factors = factors.with_columns(pl.col(factor_col).cast(pl.Float64))
    returns = returns.with_columns(pl.col(column).cast(pl.Float64) for column in columns)
    input_rows = factors.height
    exclusions = []

    def exclude(frame: pl.DataFrame, stage: str, reason: str) -> None:
        exclusions.append(
            frame.select(*keys).with_columns(
                pl.lit(stage).alias("exclusion_stage"), pl.lit(reason).alias("exclusion_reason")
            )
        )

    exclude(
        factors.filter(~pl.col(factor_col).is_finite().fill_null(False)),
        "factor",
        "missing or nonfinite factor",
    )
    factors = factors.select(*keys, factor_col).filter(pl.col(factor_col).is_finite())
    losses = {"factor": input_rows - factors.height, "returns": 0, "groups": 0, "bucketing": 0}
    factors, returns = _align_datetime_key_resolution(
        factors, returns.select(*keys, *columns), date_col
    )
    data = factors.join(returns, on=keys, how="left")
    available_returns = pl.all_horizontal(
        pl.col(column).is_finite() for column in columns
    ).fill_null(False)
    exclude(data.filter(~available_returns), "returns", "missing or nonfinite requested outcome")
    complete = data.filter(available_returns)
    losses["returns"] = data.height - complete.height
    data = complete
    group_source_hash = None
    if groups is not None:
        if isinstance(groups, Mapping):
            missing = set(factors[asset_col].unique().to_list()).difference(groups)
            if missing:
                raise KeyError(f"assets missing from group mapping: {sorted(missing, key=str)}")
            group_frame = factors.select(asset_col).unique(maintain_order=True)
            codes = [groups[asset] for asset in group_frame[asset_col].to_list()]
            group_frame = group_frame.with_columns(pl.Series("group", codes))
            group_keys = [asset_col]
        else:
            group_frame = _frame(groups, date_col, asset_col, "group").select(*keys, "group")
            data, group_frame = _align_datetime_key_resolution(data, group_frame, date_col)
            group_keys = keys
        group_source_hash = sha256(group_frame.write_ipc(None).getvalue()).hexdigest()
        if group_labels is not None and not group_frame.is_empty():
            codes = set(group_frame["group"].drop_nulls().unique().to_list())
            if missing := codes.difference(group_labels):
                raise KeyError(f"groups missing from display labels: {sorted(missing, key=str)}")
            group_frame = group_frame.with_columns(
                pl.col("group").replace_strict(dict(group_labels))
            )
        data = data.join(group_frame, on=group_keys, how="left")
        valid_group = pl.col("group").is_not_null()
        if data["group"].dtype.is_numeric():
            valid_group = valid_group & pl.col("group").is_finite()
        exclude(data.filter(~valid_group.fill_null(False)), "groups", "missing or nonfinite group")
        complete = data.filter(valid_group)
        losses["groups"] = data.height - complete.height
        data = complete
    elif binning_by_group or group_labels is not None:
        raise ValueError("group binning/labels require groups")
    bucket_keys = [date_col, "group"] if binning_by_group else [date_col]
    parts, failures = [], []
    out_of_range_rows = 0
    for part in data.sort(keys).partition_by(bucket_keys, maintain_order=True):
        try:
            assigned = PolarsBackend.fast_quantile_assignment(
                part, factor_col, quantiles, bins=bins, zero_aware=zero_aware
            )
        except (ValueError, pl.exceptions.DuplicateError) as exc:
            if max_loss == 0:
                raise
            losses["bucketing"] += part.height
            exclude(part, "bucketing", str(exc))
            key_payload = part.select(keys).write_ipc(None).getvalue()
            failures.append(
                {
                    "rows": part.height,
                    "keys_sha256": sha256(key_payload).hexdigest(),
                    "reason": str(exc),
                }
            )
            continue
        label = f"{factor_col}_quantile"
        assigned = assigned.with_columns(
            pl.col(label).cast(pl.String).cast(pl.Int32).alias("quantile")
        ).drop(label)
        retained = assigned.filter(pl.col("quantile").is_not_null())
        exclude(
            assigned.filter(pl.col("quantile").is_null()),
            "bucketing",
            "outside requested bucket edges",
        )
        out_of_range_rows += assigned.height - retained.height
        losses["bucketing"] += assigned.height - retained.height
        parts.append(retained)
    data = (
        pl.concat(parts).sort(keys)
        if parts
        else data.head(0).with_columns(pl.Series("quantile", [], dtype=pl.Int32))
    )
    loss_fraction = (input_rows - data.height) / input_rows
    if loss_fraction > max_loss:
        raise ValueError(
            f"max_loss {max_loss} exceeded by observed exclusion fraction {loss_fraction}; losses={losses}"
        )
    aligned_exclusions = []
    for excluded in exclusions:
        data, excluded = _align_datetime_key_resolution(data, excluded, date_col)
        aligned_exclusions.append(excluded)
    excluded_data = pl.concat(aligned_exclusions).sort(keys)
    payload = data.write_ipc(None).getvalue()
    metadata = {
        "input_rows": input_rows,
        "retained_rows": data.height,
        "loss_fraction": loss_fraction,
        "max_loss": float(max_loss),
        "loss_by_stage": losses,
        "excluded_data_ipc_base64": base64.b64encode(
            excluded_data.write_ipc(None).getvalue()
        ).decode("ascii"),
        "bucket_failures": failures,
        "return_columns": columns,
        "bucket_method": "pandas.cut" if bins is not None else "pandas.qcut",
        "bucket_count": bucket_count,
        "bucket_specification": specification
        if isinstance(specification, int)
        else [
            value if np.isfinite(value) else "-inf" if value < 0 else "+inf"
            for value in specification
        ],
        "infinite_edge_encoding": "-inf/+inf strings denote unbounded cut endpoints",
        "out_of_range_rows": out_of_range_rows,
        "zero_aware": bool(zero_aware),
        "binning_by_group": bool(binning_by_group),
        "frame_sha256": sha256(payload).hexdigest(),
        "input_factor_sha256": input_factor_hash,
        "input_returns_sha256": input_returns_hash,
        "computation_dtype": "Float64",
        "prepared_group_source_sha256": group_source_hash,
        "group_source": "none"
        if groups is None
        else "static"
        if isinstance(groups, Mapping)
        else "dated",
        "groups_labeled": group_labels is not None,
        "pandas_version": version("pandas"),
        "numpy_version": version("numpy"),
        "polars_version": version("polars"),
        "diagnostic_version": version("ml4t-diagnostic"),
        "sample_policy": "common finite factor/return/group sample before bucketing",
        "calendar_validation": "not performed; supplied labels do not establish endpoints or a schedule",
    }
    return PreparedSignalPanel(
        payload, json.dumps(metadata, allow_nan=False), factor_col, date_col, asset_col
    )
