"""Realistic violations: the errors a test must be shown to catch, produced the way they arise.

A test proves something only if it fails on the error it exists to prevent. That failure has to
come from input the real pipeline could emit, or from a defect in the producing code. A frame
built by hand, with a shape the pipeline cannot produce, proves nothing: a check can refuse
that frame and still miss the real error. A check that compares a column with another column
derived from the same source does the same.

Every function here takes a frame the producing code **emitted**, and changes it only the way
the world or the pipeline changes such frames:

- :func:`drop_window` removes the records of an interval, as an exchange outage, a vendor gap
  or a missing session does.
- :func:`retire_and_reuse` lists an entity under a retired identifier after a gap, as a ticker
  reused for a different firm does.
- :func:`reinsert_crossers` puts back rows the producing code emitted before a boundary filter
  that a defective filter would have kept, for example labels that exit inside a holdout.
- :func:`mutated_source` edits a producing module in place for the duration of a block, so a
  test runs against the defect it exists to catch, and restores the file afterwards.

None of them fabricates a row. Each raises :class:`ViolationNotRealizableError` when the frame
cannot carry the violation, so a test cannot pass on a perturbation that did nothing.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from ml4t.diagnostic.errors import DiagnosticError

__all__ = [
    "ViolationNotRealizableError",
    "drop_window",
    "interior_times",
    "mutated_source",
    "reinsert_crossers",
    "retire_and_reuse",
]


class ViolationNotRealizableError(DiagnosticError):
    """The emitted frame cannot carry the requested violation."""


def _require(frame: pl.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ViolationNotRealizableError(f"frame has no column(s) {missing}")


def interior_times(
    frame: pl.DataFrame,
    *,
    time_col: str,
    entity_col: str | None = None,
    entity: object | None = None,
    margin: int = 1,
) -> list[datetime]:
    """Distinct timestamps with at least ``margin`` emitted timestamps on each side.

    A gap is only a gap where records exist before and after it. Removing the first or last
    rows shortens a series without creating the condition a gap check exists for.
    """
    _require(frame, [time_col] + ([entity_col] if entity_col else []))
    rows = frame if entity_col is None else frame.filter(pl.col(entity_col) == entity)
    times = rows.get_column(time_col).unique().sort().to_list()
    return times[margin : len(times) - margin] if len(times) > 2 * margin else []


def drop_window(
    frame: pl.DataFrame,
    *,
    time_col: str,
    start: datetime,
    end: datetime,
    entity_col: str | None = None,
    entity: object | None = None,
) -> pl.DataFrame:
    """Remove the records in ``[start, end)``, for one entity or for all of them.

    For one entity this is an outage or a vendor gap. For all entities over a whole day, it is
    a session that should exist and is missing. The window must lie strictly inside the
    entity's emitted span, so records exist on both sides of the gap.
    """
    _require(frame, [time_col] + ([entity_col] if entity_col else []))
    if end <= start:
        raise ViolationNotRealizableError(f"empty window {start} .. {end}")
    scope = pl.lit(True) if entity_col is None else pl.col(entity_col) == entity
    in_window = (pl.col(time_col) >= start) & (pl.col(time_col) < end)
    rows = frame.filter(scope)
    if rows.is_empty():
        raise ViolationNotRealizableError(f"no rows for {entity_col}={entity!r}")
    first, last = rows.select(pl.col(time_col).min(), pl.col(time_col).max().alias("_last")).row(0)
    if not (first < start and end <= last):
        raise ViolationNotRealizableError(
            f"window {start} .. {end} is not inside the emitted span {first} .. {last}"
        )
    removed = frame.filter(scope & in_window).height
    if removed == 0:
        raise ViolationNotRealizableError(f"no records in {start} .. {end}; nothing to remove")
    return frame.filter(~(scope & in_window))


def retire_and_reuse(
    frame: pl.DataFrame,
    *,
    entity_col: str,
    time_col: str,
    retired: object,
    successor: object,
    at: datetime,
    gap: timedelta,
) -> pl.DataFrame:
    """Retire ``retired`` at ``at`` and list ``successor`` under its identifier after ``gap``.

    This is a ticker reused for a different firm: the retired entity's records stop, nothing
    trades under the identifier for ``gap``, and then a different entity's records appear under
    it. Rows of the retired entity from ``at`` on, and rows of the successor before
    ``at + gap``, are removed; the successor's remaining rows are relabelled. Code keyed on the
    identifier alone will read across the change of entity.
    """
    _require(frame, [entity_col, time_col])
    if retired == successor:
        raise ViolationNotRealizableError("retired and successor are the same entity")
    old = frame.filter((pl.col(entity_col) == retired) & (pl.col(time_col) < at))
    new = frame.filter((pl.col(entity_col) == successor) & (pl.col(time_col) >= at + gap))
    if old.is_empty() or new.is_empty():
        raise ViolationNotRealizableError(
            f"need records of {retired!r} before {at} and of {successor!r} after {at + gap}"
        )
    others = frame.filter(~pl.col(entity_col).is_in([retired, successor]))
    relabelled = new.with_columns(pl.lit(retired, dtype=frame.schema[entity_col]).alias(entity_col))
    return pl.concat([others, old, relabelled]).sort([entity_col, time_col])


def reinsert_crossers(
    filtered: pl.DataFrame,
    unfiltered: pl.DataFrame,
    *,
    start_col: str,
    end_col: str,
    boundary: datetime,
) -> pl.DataFrame:
    """Put back the rows that start before ``boundary`` and end at or after it.

    ``unfiltered`` is what the producing code emitted before its boundary filter and
    ``filtered`` is what it emitted after. The result is what a filter that tests only the
    start would have emitted: for labels, those whose decision precedes a holdout and whose
    exit falls inside it.
    """
    _require(unfiltered, [start_col, end_col])
    crossers = unfiltered.filter(
        (pl.col(start_col) < boundary) & (pl.col(end_col) >= boundary)
    ).select(filtered.columns)
    if crossers.is_empty():
        raise ViolationNotRealizableError(f"no emitted row starts before and ends after {boundary}")
    return pl.concat([filtered, crossers], how="vertical_relaxed")


def _forget(path: Path) -> None:
    for cached in path.parent.glob(f"__pycache__/{path.stem}.*.pyc"):
        cached.unlink(missing_ok=True)
    resolved = path.resolve()
    for name, module in list(sys.modules.items()):
        if Path(getattr(module, "__file__", "") or "").resolve() == resolved:
            del sys.modules[name]
    importlib.invalidate_caches()


@contextlib.contextmanager
def mutated_source(path: str | Path, old: str, new: str) -> Iterator[Path]:
    """Replace ``old`` with ``new`` in a producing module for the duration of the block.

    ``old`` must occur exactly once, so the mutation is the one intended. The original text is
    restored on exit, including when the block raises, and cached bytecode and imported copies
    of the module are discarded both ways, so code run inside the block sees the defect and
    code run after it does not.
    """
    path = Path(path)
    original = path.read_text(encoding="utf-8")
    count = original.count(old)
    if count != 1:
        raise ViolationNotRealizableError(f"{old!r} occurs {count} times in {path}; need exactly 1")
    path.write_text(original.replace(old, new, 1), encoding="utf-8")
    _forget(path)
    try:
        yield path
    finally:
        path.write_text(original, encoding="utf-8")
        _forget(path)
