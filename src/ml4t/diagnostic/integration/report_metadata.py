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
