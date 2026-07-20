"""Model-class-agnostic look-ahead auditor for feature extractors.

Look-ahead bias in a *model-based* feature is invisible to the eye and survives
careful-looking code. There is exactly one property that defines a causal
feature, independent of whether the underlying model is an HMM, a GARCH, a PCA,
an IPCA, an autoencoder, or an SDF:

    The feature value at ``(symbol, t)`` is a deterministic function of inputs at
    times ``<= t``. Nothing after ``t`` may change it.

This module tests that property as a **future-perturbation invariance**, without
reading the model code: destroy the inputs after a cutoff ``T``, re-run the
extractor, and assert the features at ``t <= T`` are unchanged. If any model was
fit on (or filtered over) data past ``T`` and applied backward, the past features
move and the audit fails. One mechanism catches the smoothed-HMM bug, full-sample
PCA loadings, an autoencoder trained on the whole panel, GARCH full-window
variance targeting, and a time-axis z-score - one auditor, every model class.

Two entry points share one engine:

- :func:`audit_lookahead` returns a :class:`CausalityReport` (the reader-facing
  diagnostic, renderable to HTML/JSON/Markdown).
- :func:`assert_causal` is the CI one-liner; it raises :class:`CausalityError`
  (with the report attached) on failure.

What this deliberately does **not** catch
-----------------------------------------
- **Same-timestamp label leakage** (a feature that uses the contemporaneous
  target) survives truncation - it needs a separate label-leakage check.
- **Per-fold train/val seal** - a different invariant, about the CV split rather
  than the temporal causality of the extractor.

Perturbation invariance is the uniform look-ahead gate; it is one layer, not the
whole leakage story.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from ml4t.diagnostic.errors import DiagnosticError
from ml4t.diagnostic.reporting.base import ReportFactory, ReportFormat
from ml4t.diagnostic.results.base import BaseResult

Extractor = Callable[[pl.DataFrame], pl.DataFrame]

DEFAULT_CORRUPTIONS: tuple[str, ...] = ("nan", "shuffle", "noise")
DEFAULT_KEYS: tuple[str, ...] = ("symbol", "timestamp")
DEFAULT_QUANTILES: tuple[float, ...] = (0.4, 0.6, 0.8)
_VALID_CORRUPTIONS = frozenset(DEFAULT_CORRUPTIONS)

_NUMERIC_DTYPES = (
    pl.Float32,
    pl.Float64,
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class CausalityError(DiagnosticError):
    """Raised when a feature extractor leaks future information.

    The full :class:`CausalityReport` is attached as :attr:`report` so callers
    (and CI logs) can inspect exactly which columns leaked, where, and by how
    much.

    Attributes:
        report: The :class:`CausalityReport` that triggered the failure.
    """

    def __init__(
        self,
        message: str,
        report: CausalityReport,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, context=context)
        self.report = report


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LeakEvent:
    """A single detected leak: one column exposed by one cutoff/corruption.

    Attributes:
        column: Name of the leaking feature column.
        cutoff: Cutoff timestamp whose future was corrupted.
        corruption: Corruption strategy that exposed the leak.
        first_leak: Earliest pre-cutoff timestamp whose feature value moved.
        max_abs_delta: Largest absolute change over pre-cutoff rows.
        threshold: Noise-floor-derived threshold the delta had to exceed.
    """

    column: str
    cutoff: Any
    corruption: str
    first_leak: Any
    max_abs_delta: float
    threshold: float


class _CausalityResultSchema(BaseResult):
    """Internal :class:`BaseResult` view so the ``reporting/`` renderers apply.

    Holds only JSON-serializable primitives; the leak table is reconstructed as
    a Polars DataFrame on demand in :meth:`get_dataframe`.
    """

    analysis_type: str = "lookahead_causality"
    is_causal: bool
    is_deterministic: bool
    noise_floor: float
    n_effective_probes: int
    n_skipped_probes: int
    feature_cols: list[str]
    leaking_columns: list[str]
    summary_text: str
    leak_rows: list[dict[str, Any]]

    def summary(self) -> str:
        return self.summary_text

    def list_available_dataframes(self) -> list[str]:
        return ["leaks"] if self.leak_rows else []

    def get_dataframe(self, name: str | None = None) -> pl.DataFrame:
        if not self.leak_rows:
            return pl.DataFrame()
        return pl.DataFrame(self.leak_rows)


@dataclass(frozen=True)
class CausalityReport:
    """Per-column, per-timestamp look-ahead audit result.

    Attributes:
        is_causal: True iff no feature column leaked under any requested
            corruption at any cutoff.
        leaking_columns: Mapping ``column -> {"first_leak", "max_abs_delta",
            "cutoff", "corruption"}`` for the worst (largest-delta) leak observed
            for that column. Empty when :attr:`is_causal` is True.
        determinism: ``{"is_deterministic": bool, "noise_floor": float}`` measured
            by running the extractor twice on the unperturbed frame *before* any
            perturbation. ``noise_floor`` is the max per-column noise across
            feature columns; a leak is only counted when it rises above this
            floor.
        leak_events: Every individual leak detected (column x cutoff x
            corruption), not just the per-column worst case.
        feature_cols: Feature columns that were audited.
        cutoffs: Cutoff timestamps used.
        corruptions: Corruption strategies applied.
        keys: Key columns (never corrupted).
        n_rows: Number of rows in the audited frame.
        n_effective_probes: Number of cutoff/corruption pairs that changed at
            least one future input value and were evaluated.
        n_skipped_probes: Number of requested probes skipped because the
            corruption could not change the observed future inputs.
    """

    is_causal: bool
    leaking_columns: dict[str, dict[str, Any]]
    determinism: dict[str, Any]
    leak_events: tuple[LeakEvent, ...]
    feature_cols: tuple[str, ...]
    cutoffs: tuple[Any, ...]
    corruptions: tuple[str, ...]
    keys: tuple[str, ...]
    n_rows: int = 0
    n_effective_probes: int = 0
    n_skipped_probes: int = 0

    # -- human-readable summary ------------------------------------------------
    def summary(self) -> str:
        """Return a human-readable audit summary."""
        det = self.determinism
        det_line = (
            "deterministic"
            if det.get("is_deterministic", True)
            else f"NON-deterministic (noise floor = {det.get('noise_floor', 0.0):.3e})"
        )
        lines = [
            "Look-Ahead Causality Audit",
            f"  Feature columns audited : {len(self.feature_cols)} "
            f"({', '.join(self.feature_cols) if self.feature_cols else 'none'})",
            f"  Cutoffs                 : {len(self.cutoffs)}",
            f"  Corruptions             : {', '.join(self.corruptions)}",
            f"  Effective probes        : {self.n_effective_probes} "
            f"({self.n_skipped_probes} skipped)",
            f"  Determinism             : {det_line}",
            f"  Verdict                 : {'CAUSAL' if self.is_causal else 'LEAK DETECTED'}",
        ]
        if not self.is_causal:
            lines.append("  Leaking columns:")
            for col, info in self.leaking_columns.items():
                lines.append(
                    f"    - {col}: leaks from {info['first_leak']} at "
                    f"{info['max_abs_delta']:.4g} under {info['corruption']} "
                    f"(cutoff {info['cutoff']})"
                )
        return "\n".join(lines)

    # -- renderer bridge -------------------------------------------------------
    def _to_schema(self) -> _CausalityResultSchema:
        leak_rows = [
            {
                "column": ev.column,
                "first_leak": str(ev.first_leak),
                "cutoff": str(ev.cutoff),
                "corruption": ev.corruption,
                "max_abs_delta": float(ev.max_abs_delta),
                "threshold": float(ev.threshold),
            }
            for ev in self.leak_events
        ]
        return _CausalityResultSchema(
            is_causal=self.is_causal,
            is_deterministic=bool(self.determinism.get("is_deterministic", True)),
            noise_floor=float(self.determinism.get("noise_floor", 0.0)),
            n_effective_probes=self.n_effective_probes,
            n_skipped_probes=self.n_skipped_probes,
            feature_cols=list(self.feature_cols),
            leaking_columns=list(self.leaking_columns.keys()),
            summary_text=self.summary(),
            leak_rows=leak_rows,
        )

    def to_html(self) -> str:
        """Render the report as a standalone HTML document."""
        return ReportFactory.render(self._to_schema(), ReportFormat.HTML)

    def to_markdown(self) -> str:
        """Render the report as Markdown."""
        return ReportFactory.render(self._to_schema(), ReportFormat.MARKDOWN)

    def to_json(self) -> str:
        """Render the report as a JSON string."""
        return ReportFactory.render(self._to_schema(), ReportFormat.JSON)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _numeric_cols(frame: pl.DataFrame, cols: Sequence[str]) -> list[str]:
    return [c for c in cols if frame.schema[c] in _NUMERIC_DTYPES]


def _resolve_cutoffs(
    frame: pl.DataFrame,
    time_col: str,
    cutoffs: str | Sequence[Any],
    quantiles: Sequence[float],
) -> list[Any]:
    times = frame.select(time_col).unique().sort(time_col).get_column(time_col)
    n = len(times)
    if n < 2:
        raise ValueError("look-ahead audit requires at least two distinct timestamps")
    if isinstance(cutoffs, str):
        if cutoffs != "auto":
            raise ValueError(f"cutoffs must be 'auto' or a sequence, got {cutoffs!r}")
        if not quantiles:
            raise ValueError("quantiles must be non-empty when cutoffs='auto'")
        bad_quantiles = [q for q in quantiles if not 0.0 < q < 1.0]
        if bad_quantiles:
            raise ValueError(f"quantiles must be strictly between 0 and 1, got {bad_quantiles}")
        # Inner quantiles of the time axis; drop the max so post-cutoff rows exist.
        idxs = sorted({int(round(q * (n - 1))) for q in quantiles})
        idxs = [i for i in idxs if 0 <= i < n - 1]
        if not idxs:
            idxs = [max(0, n - 2)]
        return [times[i] for i in idxs]
    resolved = list(cutoffs)
    if not resolved:
        raise ValueError("cutoffs sequence is empty")
    for cutoff in resolved:
        has_past = frame.select((pl.col(time_col) <= cutoff).any()).item()
        has_future = frame.select((pl.col(time_col) > cutoff).any()).item()
        if not has_past or not has_future:
            raise ValueError(
                f"cutoff {cutoff!r} must have at least one row on each side "
                f"({time_col} <= cutoff and {time_col} > cutoff)"
            )
    return resolved


def _partition(frame: pl.DataFrame, group_cols: Sequence[str]) -> list[pl.DataFrame]:
    if not group_cols:
        return [frame]
    return frame.partition_by(list(group_cols), maintain_order=True)


def _unused_column_name(columns: Sequence[str], base: str) -> str:
    name = base
    while name in columns:
        name += "_"
    return name


def _validate_unique_keys(frame: pl.DataFrame, keys: Sequence[str], source: str) -> None:
    key_frame = frame.select(list(keys))
    null_keys = [key for key in keys if key_frame.get_column(key).null_count()]
    if null_keys:
        raise ValueError(f"{source} contains null key values in columns: {null_keys}")
    duplicate_mask = key_frame.is_duplicated()
    if duplicate_mask.any():
        examples = key_frame.filter(duplicate_mask).unique(maintain_order=True).head(5).to_dicts()
        raise ValueError(
            f"{source} contains duplicate keys for columns {list(keys)}; examples: {examples}"
        )


def _corrupt(
    frame: pl.DataFrame,
    cutoff: Any,
    corruption: str,
    input_cols: Sequence[str],
    group_cols: Sequence[str],
    time_col: str,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Return a copy of ``frame`` with input columns destroyed for ``t > cutoff``.

    Keys are never touched. ``corruption`` is one of ``nan``, ``shuffle`` or
    ``noise``.
    """
    future = pl.col(time_col) > cutoff

    if corruption == "nan":
        return frame.with_columns(
            [pl.when(future).then(None).otherwise(pl.col(c)).alias(c) for c in input_cols]
        )

    if corruption == "shuffle":
        position_col = _unused_column_name(frame.columns, "__ml4t_causality_row_index")
        indexed = frame.with_row_index(position_col)
        pre = indexed.filter(~future)
        post = indexed.filter(future)
        if post.is_empty():
            return frame
        shuffled_parts: list[pl.DataFrame] = []
        for part in _partition(post, group_cols):
            n = len(part)
            if n == 1:
                shuffled_parts.append(
                    part.with_columns(
                        [pl.lit(None, dtype=part.schema[c]).alias(c) for c in input_cols]
                    )
                )
                continue
            perm = rng.permutation(n)
            if np.array_equal(perm, np.arange(n)):
                perm = np.roll(perm, 1)
            keys_rest = part.drop(list(input_cols))
            reordered_inputs = part.select(list(input_cols))[perm.tolist()]
            shuffled_parts.append(pl.concat([keys_rest, reordered_inputs], how="horizontal"))
        post_shuffled = pl.concat(shuffled_parts).select(indexed.columns)
        return (
            pl.concat([pre, post_shuffled])
            .sort(position_col)
            .drop(position_col)
            .select(frame.columns)
        )

    if corruption == "noise":
        numeric = _numeric_cols(frame, input_cols)
        non_numeric = [c for c in input_cols if c not in numeric]
        exprs = []
        for c in numeric:
            if group_cols:
                mean = pl.col(c).mean().over(list(group_cols))
                std = pl.col(c).std().over(list(group_cols))
            else:
                mean = pl.lit(frame.get_column(c).mean())
                std = pl.lit(frame.get_column(c).std())
            draw = pl.Series(c + "__z", rng.standard_normal(len(frame)))
            noisy = mean + draw * std.fill_null(0.0)
            exprs.append(
                pl.when(future)
                .then(noisy)
                .otherwise(pl.col(c))
                .cast(frame.schema[c], strict=False)
                .alias(c)
            )
        exprs.extend(
            [
                pl.when(future)
                .then(pl.lit(None, dtype=frame.schema[c]))
                .otherwise(pl.col(c))
                .alias(c)
                for c in non_numeric
            ]
        )
        return frame.with_columns(exprs).select(frame.columns)

    raise ValueError(f"unknown corruption strategy: {corruption!r}")


def _abs_delta_frame(
    base: pl.DataFrame,
    other: pl.DataFrame,
    col: str,
    keys: Sequence[str],
    time_col: str,
) -> pl.DataFrame:
    """Join two extractor outputs on keys and return (time, delta) for ``col``.

    ``delta`` is the absolute difference for numeric columns, or a large sentinel
    where exactly one side is null / the values differ for non-numeric columns.
    """
    selected_columns = [*keys, col]
    right_col = _unused_column_name(selected_columns, f"{col}__other")
    left_present = _unused_column_name([*selected_columns, right_col], "__left_present")
    right_present = _unused_column_name(
        [*selected_columns, right_col, left_present], "__right_present"
    )
    left = base.select(selected_columns).with_columns(pl.lit(True).alias(left_present))
    right = other.select([*keys, pl.col(col).alias(right_col)]).with_columns(
        pl.lit(True).alias(right_present)
    )
    joined = left.join(right, on=list(keys), how="full", coalesce=True)
    is_numeric = base.schema[col] in _NUMERIC_DTYPES
    row_mismatch = pl.col(left_present).is_null() | pl.col(right_present).is_null()
    null_mismatch = pl.col(col).is_null() != pl.col(right_col).is_null()
    if is_numeric:
        nan_mismatch = pl.col(col).is_nan() != pl.col(right_col).is_nan()
        raw = (pl.col(col) - pl.col(right_col)).abs().fill_nan(0.0)
        delta = (
            pl.when(row_mismatch | null_mismatch | nan_mismatch)
            .then(float("inf"))
            .otherwise(raw.fill_null(0.0))
            .alias("__delta")
        )
    else:
        differs = row_mismatch | null_mismatch | (pl.col(col) != pl.col(right_col))
        delta = pl.when(differs).then(float("inf")).otherwise(0.0).alias("__delta")
    return joined.select(pl.col(time_col), delta)


def _column_max_delta(delta_frame: pl.DataFrame) -> float:
    val = delta_frame.select(pl.col("__delta").max()).item()
    return float(val) if val is not None else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def audit_lookahead(
    extract: Extractor,
    frame: pl.DataFrame,
    cutoffs: str | Sequence[Any] = "auto",
    corruptions: Sequence[str] = DEFAULT_CORRUPTIONS,
    keys: Sequence[str] = DEFAULT_KEYS,
    feature_cols: Sequence[str] | None = None,
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    seed: int = 0,
    atol: float = 1e-9,
    noise_multiplier: float = 5.0,
) -> CausalityReport:
    """Audit a feature extractor for look-ahead bias via future-perturbation.

    The extractor is called on the unperturbed frame (twice, to measure a
    per-column determinism noise floor) and then once per ``(cutoff, corruption)``
    pair on a frame whose *input* columns are destroyed for rows after the cutoff.
    A feature column at ``t <= cutoff`` that moves by more than the noise floor is
    a leak: its value depended on the future.

    Args:
        extract: ``Callable[[pl.DataFrame], pl.DataFrame]`` mapping a
            ``(symbol, timestamp, ...)`` panel to a frame carrying ``keys`` and
            the feature columns. Output is aligned back to inputs on ``keys``.
        frame: The input panel. Must contain all ``keys``.
        cutoffs: ``"auto"`` (inner quantiles of the time axis) or an explicit
            sequence of cutoff timestamps.
        corruptions: Subset of ``("nan", "shuffle", "noise")``. A column is
            declared causal only if invariant under *all* requested corruptions.
        keys: Key columns. The last key is the time axis; the rest are entity
            (e.g. ``symbol``) groups. Keys are never corrupted.
        feature_cols: Columns to audit. Defaults to columns present in the
            extractor output but absent from ``frame`` (i.e. the derived
            features); falls back to all non-key output columns.
        quantiles: Inner quantiles used when ``cutoffs="auto"``.
        seed: Seed for the corruption RNG (shuffle/noise), for reproducibility.
        atol: Absolute floor for every leak threshold, including when a small
            non-zero determinism noise floor is measured.
        noise_multiplier: Safety factor applied to the measured per-column noise
            floor when judging a leak (``delta > noise_floor * noise_multiplier``).
            Values > 1 keep nondeterministic extractors from false-positiving on
            their own jitter; the raw noise floor is still reported unscaled.

    Returns:
        A :class:`CausalityReport`.

    Raises:
        ValueError: If validation fails, a requested probe cannot modify future
            inputs, or the extractor output cannot be aligned on unique ``keys``.
    """
    keys = tuple(keys)
    corruptions = tuple(corruptions)
    if not keys:
        raise ValueError("keys must be non-empty")
    missing_keys = [k for k in keys if k not in frame.columns]
    if missing_keys:
        raise ValueError(f"frame is missing key columns: {missing_keys}")
    _validate_unique_keys(frame, keys, "frame")
    bad = [c for c in corruptions if c not in _VALID_CORRUPTIONS]
    if bad:
        raise ValueError(
            f"unknown corruption strategies: {bad}; valid: {sorted(_VALID_CORRUPTIONS)}"
        )
    if not corruptions:
        raise ValueError("corruptions must be non-empty")
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError(f"atol must be a finite non-negative value, got {atol!r}")
    if not np.isfinite(noise_multiplier) or noise_multiplier <= 0.0:
        raise ValueError(
            f"noise_multiplier must be a finite positive value, got {noise_multiplier!r}"
        )

    time_col = keys[-1]
    group_cols = list(keys[:-1])
    input_cols = [c for c in frame.columns if c not in keys]
    if not input_cols:
        raise ValueError("frame must contain at least one non-key input column to corrupt")
    resolved_cutoffs = _resolve_cutoffs(frame, time_col, cutoffs, quantiles)

    # -- reference runs (also the determinism probe) --------------------------
    base_out = _align_output(extract(frame), frame, keys)
    base_out_2 = _align_output(extract(frame), frame, keys)
    if base_out.is_empty():
        raise ValueError("extractor output is empty; no feature rows are available to audit")
    _validate_unique_keys(base_out, keys, "extractor output")
    _validate_unique_keys(base_out_2, keys, "second extractor output")

    # -- resolve feature columns ---------------------------------------------
    if feature_cols is None:
        derived = [c for c in base_out.columns if c not in frame.columns]
        resolved_features = derived or [c for c in base_out.columns if c not in keys]
    else:
        resolved_features = list(feature_cols)
    if not resolved_features:
        raise ValueError("extractor output contains no feature columns to audit")
    key_features = [col for col in resolved_features if col in keys]
    if key_features:
        raise ValueError(f"feature columns must not overlap key columns, got: {key_features}")
    missing_feats = [c for c in resolved_features if c not in base_out.columns]
    if missing_feats:
        raise ValueError(f"extractor output is missing feature columns: {missing_feats}")
    missing_second = [c for c in resolved_features if c not in base_out_2.columns]
    if missing_second:
        raise ValueError(f"second extractor output is missing feature columns: {missing_second}")

    # -- per-column noise floor ----------------------------------------------
    noise_floor: dict[str, float] = {}
    for col in resolved_features:
        df = _abs_delta_frame(base_out, base_out_2, col, keys, time_col)
        noise_floor[col] = _column_max_delta(df)
    non_finite_floors = [col for col, floor in noise_floor.items() if not np.isfinite(floor)]
    if non_finite_floors:
        raise ValueError(
            "non-finite determinism noise floor for feature columns "
            f"{non_finite_floors}; extractor row, null, or NaN output changed between base runs"
        )
    max_floor = max(noise_floor.values(), default=0.0)
    is_deterministic = max_floor <= 0.0

    thresholds = {col: max(fl * noise_multiplier, atol) for col, fl in noise_floor.items()}

    # -- perturbation sweep ---------------------------------------------------
    leak_events: list[LeakEvent] = []
    comparison_counts = dict.fromkeys(resolved_features, 0)
    n_effective_probes = 0
    n_skipped_probes = 0
    counter = 0
    for cutoff in resolved_cutoffs:
        for corruption in corruptions:
            counter += 1
            rng = np.random.default_rng(seed + counter)
            corrupted = _corrupt(frame, cutoff, corruption, input_cols, group_cols, time_col, rng)
            if frame.equals(corrupted):
                n_skipped_probes += 1
                continue
            n_effective_probes += 1
            pert_out = _align_output(extract(corrupted), corrupted, keys)
            _validate_unique_keys(
                pert_out,
                keys,
                f"extractor output for {corruption} corruption at cutoff {cutoff!r}",
            )
            missing_perturbed = [c for c in resolved_features if c not in pert_out.columns]
            if missing_perturbed:
                raise ValueError(
                    f"perturbed extractor output is missing feature columns: {missing_perturbed}"
                )
            for col in resolved_features:
                df = _abs_delta_frame(base_out, pert_out, col, keys, time_col)
                df = df.filter(pl.col(time_col) <= cutoff)
                if df.is_empty():
                    continue
                comparison_counts[col] += len(df)
                max_delta = _column_max_delta(df)
                threshold = thresholds[col]
                if max_delta > threshold:
                    first = (
                        df.filter(pl.col("__delta") > threshold)
                        .select(pl.col(time_col).min())
                        .item()
                    )
                    leak_events.append(
                        LeakEvent(
                            column=col,
                            cutoff=cutoff,
                            corruption=corruption,
                            first_leak=first,
                            max_abs_delta=max_delta,
                            threshold=threshold,
                        )
                    )

    if n_effective_probes == 0:
        raise ValueError(
            "all requested corruption probes were ineffective for the observed future inputs"
        )
    never_compared = [col for col, count in comparison_counts.items() if count == 0]
    if never_compared:
        raise ValueError(
            f"no pre-cutoff comparisons were made for feature columns: {never_compared}"
        )

    # -- aggregate per column (worst leak) ------------------------------------
    leaking_columns: dict[str, dict[str, Any]] = {}
    for ev in leak_events:
        current = leaking_columns.get(ev.column)
        if current is None or ev.max_abs_delta > current["max_abs_delta"]:
            leaking_columns[ev.column] = {
                "first_leak": ev.first_leak,
                "max_abs_delta": ev.max_abs_delta,
                "cutoff": ev.cutoff,
                "corruption": ev.corruption,
            }

    return CausalityReport(
        is_causal=len(leaking_columns) == 0,
        leaking_columns=leaking_columns,
        determinism={"is_deterministic": is_deterministic, "noise_floor": max_floor},
        leak_events=tuple(leak_events),
        feature_cols=tuple(resolved_features),
        cutoffs=tuple(resolved_cutoffs),
        corruptions=corruptions,
        keys=keys,
        n_rows=len(frame),
        n_effective_probes=n_effective_probes,
        n_skipped_probes=n_skipped_probes,
    )


def assert_causal(
    extract: Extractor,
    frame: pl.DataFrame,
    cutoffs: str | Sequence[Any] = "auto",
    corruptions: Sequence[str] = DEFAULT_CORRUPTIONS,
    keys: Sequence[str] = DEFAULT_KEYS,
    feature_cols: Sequence[str] | None = None,
    **kwargs: Any,
) -> CausalityReport:
    """Run :func:`audit_lookahead` and raise if any feature leaks.

    Args:
        extract: See :func:`audit_lookahead`.
        frame: See :func:`audit_lookahead`.
        cutoffs: See :func:`audit_lookahead`.
        corruptions: See :func:`audit_lookahead`.
        keys: See :func:`audit_lookahead`.
        feature_cols: See :func:`audit_lookahead`.
        **kwargs: Forwarded to :func:`audit_lookahead` (``seed``, ``atol``,
            ``noise_multiplier``, ``quantiles``).

    Returns:
        The :class:`CausalityReport` on success (all features causal).

    Raises:
        CausalityError: If any feature column leaks; the report is attached as
            :attr:`CausalityError.report`.
    """
    report = audit_lookahead(
        extract,
        frame,
        cutoffs=cutoffs,
        corruptions=corruptions,
        keys=keys,
        feature_cols=feature_cols,
        **kwargs,
    )
    if not report.is_causal:
        cols = ", ".join(report.leaking_columns)
        raise CausalityError(
            f"Look-ahead leak detected in feature column(s): {cols}",
            report=report,
            context={"leaking_columns": list(report.leaking_columns)},
        )
    return report


def _align_output(
    output: pl.DataFrame,
    frame: pl.DataFrame,
    keys: Sequence[str],
) -> pl.DataFrame:
    """Ensure the extractor output carries ``keys`` for join-based comparison.

    If keys are absent but the output is row-aligned with the input frame, the
    keys are attached from the input by position.
    """
    if all(k in output.columns for k in keys):
        return output
    if len(output) == len(frame):
        return output.with_columns([frame.get_column(k) for k in keys])
    raise ValueError(
        "extractor output does not contain key columns "
        f"{list(keys)} and is not row-aligned with the input frame "
        f"(output rows={len(output)}, input rows={len(frame)})"
    )


__all__ = [
    "CausalityError",
    "CausalityReport",
    "LeakEvent",
    "audit_lookahead",
    "assert_causal",
]
