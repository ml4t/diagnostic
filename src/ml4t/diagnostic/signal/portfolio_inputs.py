"""Daily analytical interchange from supplied outcomes and synthetic allocations."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from io import BytesIO
from numbers import Real
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import polars as pl

from ml4t.diagnostic.signal._utils import _signal_frame
from ml4t.diagnostic.signal.portfolio import compute_factor_cumulative_returns
from ml4t.diagnostic.signal.positions import SyntheticPositionsResult
from ml4t.diagnostic.utils.validation import (
    _align_datetime_key_resolution,
    _validate_observation_keys,
)

if TYPE_CHECKING:
    from ml4t.diagnostic.evaluation.portfolio_analysis import PortfolioAnalysis


@dataclass(frozen=True)
class SyntheticPortfolioInputs:
    """Owned daily returns, holdings and valuations for a research portfolio.

    Return paths compound supplied forward outcomes, not marked account P&L.
    Values are portfolio fractions unless capital was supplied. A missing NAV
    beyond the observed return path remains unknown, never invented as zero.
    Cash is stored separately from asset IDs so an asset named cash is valid.
    """

    _returns: bytes = field(repr=False)
    _positions: bytes = field(repr=False)
    _valuations: bytes = field(repr=False)
    _metadata: str = field(repr=False)
    _observations: bytes = field(repr=False)
    _benchmark: bytes | None = field(default=None, repr=False)

    @property
    def returns(self) -> pl.DataFrame:
        """Return independent dated daily returns, wealth and optional benchmark."""
        return pl.read_ipc(BytesIO(self._returns))

    @property
    def positions(self) -> pl.DataFrame:
        """Return independent long asset weights and fraction/currency values."""
        return pl.read_ipc(BytesIO(self._positions))

    @property
    def valuations(self) -> pl.DataFrame:
        """Return independent daily cash weights, NAV and cash values."""
        return pl.read_ipc(BytesIO(self._valuations))

    @property
    def observations(self) -> pl.DataFrame:
        """Return original dated horizon returns, finite counts and wealth before resampling."""
        return pl.read_ipc(BytesIO(self._observations))

    @property
    def benchmark(self) -> pl.DataFrame | None:
        """Return the independent full-universe daily benchmark clock, when supplied."""
        return pl.read_ipc(BytesIO(self._benchmark)) if self._benchmark is not None else None

    @property
    def metadata(self) -> dict:
        """Return independent definitions, units and input provenance."""
        return json.loads(self._metadata)

    def to_json(self) -> str:
        """Serialize typed Arrow snapshots without nonstandard JSON numbers."""
        return json.dumps(
            {
                "format_version": 1,
                "tables": {
                    name: base64.b64encode(getattr(self, f"_{name}")).decode("ascii")
                    for name in ("returns", "positions", "valuations", "observations")
                },
                "metadata": self.metadata,
                "benchmark_ipc_base64": base64.b64encode(self._benchmark).decode("ascii")
                if self._benchmark is not None
                else None,
            },
            allow_nan=False,
        )

    @classmethod
    def from_json(cls, value: str) -> SyntheticPortfolioInputs:
        """Restore the versioned result and verify its Arrow tables can be read."""
        parsed = json.loads(value)
        if parsed.get("format_version") != 1:
            raise ValueError("unsupported synthetic portfolio inputs format_version")
        tables = [
            base64.b64decode(parsed["tables"][name], validate=True)
            for name in ("returns", "positions", "valuations", "observations")
        ]
        for payload in tables:
            pl.read_ipc(BytesIO(payload))
        benchmark = (
            base64.b64decode(parsed["benchmark_ipc_base64"], validate=True)
            if parsed.get("benchmark_ipc_base64") is not None
            else None
        )
        if benchmark is not None:
            pl.read_ipc(BytesIO(benchmark))
        return cls(
            tables[0],
            tables[1],
            tables[2],
            json.dumps(parsed["metadata"], allow_nan=False),
            tables[3],
            benchmark,
        )

    def to_pyfolio(self) -> tuple[pd.Series, pd.DataFrame, pd.Series | None]:
        """Return independent legacy-shaped returns, positions including cash, benchmark.

        Pyfolio's wide representation reserves the column cash. Use the native
        long tables when an actual asset has that ID. No transactions are created.
        """
        returns = self.returns.to_pandas().set_index("date")
        positions = self.positions.to_pandas().pivot(index="date", columns="asset", values="value")
        returns.index.freq = "D"
        positions.index.freq = "D"
        if "cash" in positions.columns:
            raise ValueError(
                "the Pyfolio wide format reserves asset ID 'cash'; use native long tables"
            )
        positions.columns.name = "asset"
        positions["cash"] = self.valuations.to_pandas().set_index("date")["cash"]
        benchmark_frame = self.benchmark
        benchmark = (
            benchmark_frame.to_pandas().set_index("date")["benchmark"]
            if benchmark_frame is not None
            else None
        )
        if benchmark is not None:
            benchmark.index.freq = "D"
        supplied_returns = returns["return"].rename(None)
        provenance = {
            "representation": "synthetic_daily_portfolio_inputs",
            "snapshot_kind": "synthetic",
            "position_unit": self.metadata["position_unit"],
            "return_definition": "compounded supplied forward outcomes, not executed account P&L",
            "transactions": "not_created",
        }
        for frame in (supplied_returns, positions, benchmark):
            if frame is not None:
                frame.attrs["diagnostic"] = provenance.copy()
        return supplied_returns, positions, benchmark

    def analyze(self, *, periods_per_year: int, risk_free: float = 0.0) -> PortfolioAnalysis:
        """Analyze daily returns at the caller's explicit annualization scale.

        This delegates to PortfolioAnalysis, not a second report/metric engine.
        Mathematical allocations are not relabelled executed holdings or fills.
        """
        from ml4t.diagnostic.evaluation.portfolio_analysis import PortfolioAnalysis

        returns = self.returns
        return PortfolioAnalysis(
            returns=returns["return"],
            dates=returns["date"],
            benchmark=returns["benchmark"] if "benchmark" in returns.columns else None,
            periods_per_year=periods_per_year,
            risk_free=risk_free,
        )


def _daily_wealth(path: pl.DataFrame) -> pd.Series:
    if path.is_empty():
        raise ValueError("portfolio interchange requires at least one selected factor date")
    wealth = path.select("date", "wealth").to_pandas().set_index("date")["wealth"]
    wealth.index = pd.DatetimeIndex(wealth.index)
    return wealth.sort_index().resample("1D").last().ffill()


def _daily_returns(wealth: pd.Series, first_return: str) -> pd.Series:
    returns = wealth.pct_change(fill_method=None).fillna(0.0)
    if first_return == "from_initial_wealth" and len(returns):
        returns.iloc[0] = wealth.iloc[0] - 1.0
    return returns


def build_factor_portfolio_inputs(
    data: pl.DataFrame | pd.DataFrame,
    *,
    positions: SyntheticPositionsResult,
    return_column: str,
    benchmark_column: str | None = "1D",
    capital: float | None = None,
    position_method: Literal["end_of_day", "legacy_event_sum"] = "end_of_day",
    first_return: Literal["from_initial_wealth", "zero"] = "from_initial_wealth",
) -> SyntheticPortfolioInputs:
    """Compose shared factor returns, event holdings and explicit daily conventions.

    The allocation result determines the selected universe and weighting flags.
    data supplies the matching factors/outcomes, including the original full
    benchmark universe. Daily end-of-day holdings carry the last active weights;
    legacy_event_sum instead sums union-event weights within a day, including
    zero-valued empty daily bins, then renormalizes. Both are analytical views,
    not account execution. Civil-day resampling includes weekends and does not
    validate exchange schedules. Capital must be finite and strictly positive.
    """
    if position_method not in ("end_of_day", "legacy_event_sum"):
        raise ValueError("position_method must be end_of_day or legacy_event_sum")
    if first_return not in ("from_initial_wealth", "zero"):
        raise ValueError("first_return must be from_initial_wealth or zero")
    if capital is not None and (
        isinstance(capital, bool)
        or not isinstance(capital, Real)
        or not np.isfinite(capital)
        or capital <= 0
    ):
        raise ValueError("capital must be finite and strictly positive")
    metadata = positions.metadata
    contract = metadata["input_contract"]
    date_col, asset_col = contract["timestamp_col"], contract["entity_col"]
    factor_col = metadata.get("factor_col", "factor")
    group_col = metadata.get("group_col", "group")
    values = _signal_frame(data).rename(
        {date_col: "date", asset_col: "asset", factor_col: "factor"}
    )
    if group_col in values.columns and group_col != "group":
        values = values.rename({group_col: "group"})
    allocations = positions.allocations
    _validate_observation_keys(values, ["date", "asset"], "portfolio input")
    # The owned allocation sample is authoritative, including its original factors.
    outcomes = values.select("date", "asset", return_column)
    allocations, outcomes = _align_datetime_key_resolution(allocations, outcomes, "date")
    if allocations.join(outcomes.select("date", "asset"), on=["date", "asset"], how="anti").height:
        raise ValueError(
            "portfolio data must retain every selected allocation key; use explicit null outcomes for unknown values"
        )
    selected = (
        allocations.drop(return_column) if return_column in allocations.columns else allocations
    )
    selected = selected.join(outcomes, on=["date", "asset"], how="left", validate="1:1")
    flags = {key: metadata[key] for key in ("demeaned", "group_adjust", "equal_weight")}
    path = compute_factor_cumulative_returns(selected, return_column=return_column, **flags)
    wealth = _daily_wealth(path)
    daily = pd.DataFrame({"return": _daily_returns(wealth, first_return), "wealth": wealth})
    benchmark_frame = None
    if benchmark_column is not None and benchmark_column in values.columns:
        benchmark = compute_factor_cumulative_returns(
            values.with_columns(pl.col("factor").abs()),
            return_column=benchmark_column,
            demeaned=False,
            group_adjust=False,
            equal_weight=True,
        )
        benchmark_daily = _daily_returns(_daily_wealth(benchmark), first_return).rename("benchmark")
        benchmark_daily.index.name = "date"
        benchmark_frame = pl.from_pandas(benchmark_daily.reset_index())
        daily["benchmark"] = benchmark_daily

    assets = positions.positions["asset"].unique(maintain_order=True)
    # Integer columns avoid coercing opaque string/numeric asset IDs during resampling.
    encoded = (
        positions.positions.with_columns(
            pl.col("asset").replace_strict(assets, pl.Series(range(len(assets)))).alias("asset")
        )
        .to_pandas()
        .pivot(index="date", columns="asset", values="weight")
    )
    encoded.index = pd.DatetimeIndex(encoded.index)
    if position_method == "legacy_event_sum":
        weights = encoded.resample("1D").sum().ffill()
    else:
        weights = encoded.resample("1D").last().ffill()
    weights = weights.div(weights.abs().sum(axis=1), axis=0).fillna(0.0)
    cash_weight = 1.0 - weights.sum(axis=1)
    nav = (
        wealth.reindex(weights.index) * float(capital)
        if capital is not None
        else pd.Series(1.0, index=weights.index)
    )
    with np.errstate(over="ignore", invalid="ignore"):
        values_daily = weights.mul(nav, axis=0)
        cash = cash_weight * nav
    if np.isinf(values_daily.to_numpy()).any() or np.isinf(cash.to_numpy()).any():
        raise ValueError("capital-scaled portfolio values overflow finite representation")
    dates = pl.from_pandas(weights.index.rename("date"))
    count = len(weights)
    position_frame = pl.DataFrame(
        {
            "date": dates.gather(pl.Series(np.repeat(np.arange(count), len(assets)))),
            "asset": assets.gather(pl.Series(np.tile(np.arange(len(assets)), count))),
            "weight": weights.to_numpy().reshape(-1),
            "value": values_daily.to_numpy().reshape(-1),
        }
    ).with_columns(pl.col("value").fill_nan(None))
    valuations = pl.DataFrame(
        {
            "date": dates,
            "cash_weight": cash_weight.to_numpy(),
            "equity": nav.to_numpy(),
            "cash": cash.to_numpy(),
        }
    ).with_columns(pl.col("equity", "cash").fill_nan(None))
    daily.index.name = "date"
    daily_frame = pl.from_pandas(daily.reset_index())
    definitions = {
        **metadata,
        "representation": "synthetic_daily_portfolio_inputs",
        "position_method": position_method,
        "first_return": first_return,
        "return_column": return_column,
        "benchmark_column": benchmark_column if "benchmark" in daily.columns else None,
        "capital": None if capital is None else float(capital),
        "position_unit": "portfolio_fraction" if capital is None else "currency",
        "return_clock": "civil_day_last_wealth_ffill",
        "transactions": "not_created",
        "return_definition": "compounded supplied forward outcomes, not self-financing account P&L",
        "missing_return_policy": "flat_wealth; original nulls/counts retained in observations",
    }
    return SyntheticPortfolioInputs(
        daily_frame.write_ipc(None).getvalue(),
        position_frame.write_ipc(None).getvalue(),
        valuations.write_ipc(None).getvalue(),
        json.dumps(definitions, allow_nan=False),
        path.write_ipc(None).getvalue(),
        benchmark_frame.write_ipc(None).getvalue() if benchmark_frame is not None else None,
    )
