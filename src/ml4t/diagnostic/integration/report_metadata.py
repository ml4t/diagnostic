"""Structured metadata for backtest tear sheets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestReportMetadata:
    """Optional descriptive metadata for tearsheet presentation."""

    title: str | None = None
    subtitle: str | None = None
    strategy_name: str | None = None
    strategy_id: str | None = None
    universe: str | None = None
    benchmark_name: str | None = None
    evaluation_window: str | None = None
    run_id: str | None = None
    library_version: str | None = None
    calendar: str | None = None
    execution_summary: str | None = None
    cost_summary: str | None = None
    data_summary: str | None = None
    ml_summary: str | None = None

    def resolve_title(self) -> str:
        """Return the visible report title without inventing one."""
        return (self.title or self.strategy_name or "").strip()

    def resolve_subtitle(self) -> str:
        """Return the visible report subtitle."""
        return (self.subtitle or "").strip()

    def resolve_benchmark_name(self, fallback: str = "Benchmark") -> str:
        """Return the benchmark label for comparison surfaces."""
        value = (self.benchmark_name or "").strip()
        return value or fallback

    def merged_with(self, fallback: BacktestReportMetadata | None) -> BacktestReportMetadata:
        """Return metadata with missing fields filled from a fallback."""
        if fallback is None:
            return self
        data = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            data[field_name] = value if value not in (None, "") else getattr(fallback, field_name)
        return BacktestReportMetadata(**data)
