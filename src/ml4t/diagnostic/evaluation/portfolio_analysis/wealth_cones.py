"""Training-only portfolio wealth forecasts over explicitly counted observations."""

from __future__ import annotations

import base64
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from io import BytesIO
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars as pl
from ml4t.specs import UNSET_CALENDAR, CalendarUnset

from ..stats.bootstrap import _stationary_bootstrap_indices
from .inputs import positive_integer
from .metrics import _safe_cumprod
from .regimes import _window_mask

if TYPE_CHECKING:
    from .analysis import PortfolioAnalysis

_TABLES = ("training", "live", "paths", "bands")


@dataclass(frozen=True)
class WealthConeResult:
    """Owned training rows, simulated products, distribution bands and actual live path.

    Step one is the first forward return, not an invented future timestamp.
    Quantiles and mean plus/minus population standard deviation are separately
    labeled definitions. Neither is a confidence interval for a fitted mean.
    """

    _tables: tuple[bytes, ...] = field(repr=False)
    _metadata: str = field(repr=False)

    def _table(self, name: str) -> pl.DataFrame:
        return pl.read_ipc(BytesIO(self._tables[_TABLES.index(name)]))

    @property
    def training(self) -> pl.DataFrame:
        """Actual dated pre-live observations, with optional aligned benchmark."""
        return self._table("training")

    @property
    def live(self) -> pl.DataFrame:
        """Actual dated live observations up to the requested observation horizon."""
        return self._table("live")

    @property
    def paths(self) -> pl.DataFrame:
        """Each sampled source row, return and resulting capital-valued wealth."""
        return self._table("paths")

    @property
    def bands(self) -> pl.DataFrame:
        """Long step/definition/level/wealth table, without silently clipping bounds."""
        return self._table("bands")

    @property
    def metadata(self) -> dict[str, Any]:
        """Independent method, seed, units and known-at declarations."""
        return json.loads(self._metadata)

    def to_json(self) -> str:
        """Serialize owned exact Arrow clocks and method declarations."""
        return json.dumps(
            {
                "format": "ml4t.diagnostic.wealth_cones.v1",
                "metadata": self.metadata,
                "tables": {
                    name: base64.b64encode(data).decode("ascii")
                    for name, data in zip(_TABLES, self._tables, strict=True)
                },
            },
            allow_nan=False,
        )

    @classmethod
    def from_json(cls, value: str) -> WealthConeResult:
        """Restore a result without resampling or changing observation keys."""
        parsed = json.loads(value)
        if parsed["format"] != "ml4t.diagnostic.wealth_cones.v1":
            raise ValueError("unsupported wealth cone result format")
        tables = tuple(base64.b64decode(parsed["tables"][name], validate=True) for name in _TABLES)
        for data in tables:
            pl.read_ipc(BytesIO(data))
        return cls(tables, json.dumps(parsed["metadata"], allow_nan=False))


def _levels(values: Sequence[float], name: str, *, quantile: bool = False) -> list[float]:
    result = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{name} must contain finite real levels")
        level = float(value)
        if not np.isfinite(level) or (not 0 <= level <= 1 if quantile else level <= 0):
            raise ValueError(f"{name} contains an invalid level")
        result.append(level)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate levels")
    return result


def _sample_indices(
    n: int,
    horizon: int,
    n_samples: int,
    *,
    method: str,
    seed: int | None,
    block_size: float | None,
    legacy_rng: bool = False,
) -> np.ndarray:
    if legacy_rng:
        return np.random.RandomState(seed).choice(n, size=(n_samples, horizon))
    rng = np.random.default_rng(seed)
    if method == "iid":
        return rng.integers(0, n, size=(n_samples, horizon))
    return np.stack(
        [
            _stationary_bootstrap_indices(n, float(block_size), rng, sample_size=horizon)
            for _ in range(n_samples)
        ]
    )


def _wealth_paths(samples: np.ndarray, initial_capital: float) -> np.ndarray:
    if samples.size == 0:
        return np.empty_like(samples, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.stack([_safe_cumprod(1 + row) * initial_capital for row in samples])
    if not np.isfinite(result).all():
        raise ValueError("wealth paths overflow at the requested horizon/capital")
    return result


def analyze_wealth_cones(
    analysis: PortfolioAnalysis,
    *,
    live_start: date | datetime | str,
    horizon: int,
    n_samples: int = 1000,
    method: Literal["iid", "stationary"] = "iid",
    seed: int | None = 0,
    initial_capital: float = 1.0,
    block_size: float | None = None,
    quantiles: Sequence[float] = (0.025, 0.5, 0.975),
    cone_std: Sequence[float] = (1.0, 1.5, 2.0),
    horizon_unit: Literal["observations"] = "observations",
    calendar: str | CalendarUnset | None = UNSET_CALENDAR,
) -> WealthConeResult:
    """Forecast from rows strictly before live_start using existing product/resampling kernels.

    Observation steps do not imply civil days or exchange sessions. The resolved
    calendar is retained as provenance and never used to invent a future clock.
    IID uses a local NumPy Generator; stationary resampling uses circular geometric
    blocks and requires an explicit expected block length, in training observations.
    Benchmark draws share the same source rows, retaining contemporaneous pairing.
    """
    horizon = positive_integer(horizon, "horizon")
    n_samples = positive_integer(n_samples, "n_samples")
    if live_start is None:
        raise ValueError("live_start requires a nonmissing date or instant")
    if horizon_unit != "observations":
        raise ValueError("horizon_unit must be 'observations'; no future session clock is inferred")
    if method not in ("iid", "stationary"):
        raise ValueError("method must be 'iid' or 'stationary'")
    if seed is not None and (
        isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0
    ):
        raise ValueError("seed must be a nonnegative integer or None")
    if (
        isinstance(initial_capital, (bool, np.bool_))
        or not isinstance(initial_capital, Real)
        or not np.isfinite(initial_capital)
        or initial_capital <= 0
    ):
        raise ValueError("initial_capital must be finite and positive")
    if method == "stationary":
        if (
            isinstance(block_size, (bool, np.bool_))
            or not isinstance(block_size, Real)
            or not np.isfinite(block_size)
            or block_size < 1
        ):
            raise ValueError("stationary block_size must be finite and at least one observation")
    elif block_size is not None:
        raise ValueError("block_size is only defined for stationary resampling")
    probabilities = _levels(quantiles, "quantiles", quantile=True)
    deviations = _levels(cone_std, "cone_std")
    analysis.require_dates()
    observations = pl.DataFrame({"date": analysis.dates, "return": analysis.returns})
    if analysis.benchmark is not None:
        observations = observations.with_columns(pl.Series("benchmark", analysis.benchmark))
    pre_live = _window_mask(observations["date"], None, live_start, end_exclusive=True)
    training = observations.filter(pre_live)
    live = observations.filter(~pre_live).head(horizon)
    if training.is_empty():
        raise ValueError("wealth cones require at least one pre-live observation")
    indices = _sample_indices(
        training.height, horizon, n_samples, method=method, seed=seed, block_size=block_size
    )
    sampled = training["return"].to_numpy()[indices]
    wealth = _wealth_paths(sampled, float(initial_capital))
    paths = pl.DataFrame(
        {
            "sample": np.repeat(np.arange(n_samples), horizon),
            "step": np.tile(np.arange(1, horizon + 1), n_samples),
            "training_row": indices.ravel(),
            "sampled_return": sampled.ravel(),
            "wealth": wealth.ravel(),
        }
    )
    live = live.with_columns(
        pl.Series("step", np.arange(1, live.height + 1)),
        pl.Series("wealth", _safe_cumprod(1 + live["return"].to_numpy()) * initial_capital),
    )
    if "benchmark" in training.columns:
        relative = training["benchmark"].to_numpy()[indices]
        paths = paths.with_columns(
            pl.Series("sampled_benchmark", relative.ravel()),
            pl.Series("benchmark_wealth", _wealth_paths(relative, float(initial_capital)).ravel()),
        )
        live = live.with_columns(
            pl.Series(
                "benchmark_wealth",
                _safe_cumprod(1 + live["benchmark"].to_numpy()) * initial_capital,
            )
        )
    records = []
    with np.errstate(over="ignore", invalid="ignore"):
        means, stds = wealth.mean(axis=0), wealth.std(axis=0, ddof=0)
        for step in range(horizon):
            records.append((step + 1, "mean", 0.0, means[step]))
            records.extend(
                (step + 1, "quantile", q, np.quantile(wealth[:, step], q, method="linear"))
                for q in probabilities
            )
            records.extend(
                (step + 1, "mean_std", sign * level, means[step] + sign * level * stds[step])
                for level in deviations
                for sign in (1, -1)
            )
    bands = pl.DataFrame(
        records,
        schema={
            "step": pl.Int64,
            "definition": pl.String,
            "level": pl.Float64,
            "wealth": pl.Float64,
        },
        orient="row",
    )
    if not bands["wealth"].is_finite().all():
        raise ValueError("wealth cone summaries overflow at the requested horizon/capital")
    account = analysis.input_data.metadata if analysis.input_data is not None else {}
    resolved_calendar = account.get("calendar", "NYSE") if calendar is UNSET_CALENDAR else calendar
    if resolved_calendar is not None and (
        not isinstance(resolved_calendar, str) or not resolved_calendar.strip()
    ):
        raise ValueError("calendar must be a nonempty exchange name or None")
    metadata = {
        "method": method,
        "rng": "numpy.Generator.PCG64",
        "seed": None if seed is None else int(seed),
        "block_size": None if block_size is None else float(block_size),
        "block_size_unit": "training observations" if method == "stationary" else None,
        "initial_capital": float(initial_capital),
        "horizon": horizon,
        "horizon_unit": horizon_unit,
        "n_samples": n_samples,
        "n_training": training.height,
        "n_live": live.height,
        "live_start": str(live_start),
        "training_boundary": "date < live_start",
        "live_boundary": "date >= live_start",
        "calendar": resolved_calendar,
        "calendar_source": account.get("calendar_source", "default")
        if calendar is UNSET_CALENDAR
        else "argument",
        "calendar_application": "provenance only; observation horizon, no resampling or inferred future dates",
        "quantiles": probabilities,
        "quantile_method": "linear",
        "cone_std": deviations,
        "std_ddof": 0,
        "wealth_unit": "initial capital units",
        "return_unit": "decimal simple return",
        "benchmark_sampling": "paired training rows" if "benchmark" in training.columns else None,
        "account": account,
    }
    tables = (training, live, paths, bands)
    return WealthConeResult(
        tuple(frame.write_ipc(None).getvalue() for frame in tables),
        json.dumps(metadata, allow_nan=False),
    )
