#!/usr/bin/env python3
"""Generate a human-readable benchmark summary from pytest-benchmark JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _benchmark_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    benchmarks = payload.get("benchmarks", [])
    return {str(item.get("fullname") or item.get("name")): item for item in benchmarks}


def _median(item: dict[str, Any]) -> float:
    stats = item.get("stats", {})
    return float(stats.get("median", stats.get("mean", 0.0)))


def _format_seconds(value: float) -> str:
    if value < 1e-3:
        return f"{value * 1e6:.2f} us"
    if value < 1:
        return f"{value * 1e3:.2f} ms"
    return f"{value:.3f} s"


def generate_report(
    current_json: Path,
    output_md: Path,
    baseline_json: Path | None = None,
    regression_threshold: float = 1.25,
) -> None:
    current = _load(current_json)
    current_map = _benchmark_map(current)
    baseline_map: dict[str, dict[str, Any]] = {}
    if baseline_json is not None and baseline_json.exists():
        baseline_map = _benchmark_map(_load(baseline_json))

    lines: list[str] = []
    lines.append("# Benchmark Report")
    lines.append("")
    lines.append(f"- Current file: `{current_json}`")
    if baseline_json is not None:
        lines.append(
            f"- Baseline file: `{baseline_json}` ({'found' if baseline_json.exists() else 'not found'})"
        )
    lines.append("")

    if not current_map:
        lines.append("No benchmark entries found.")
        output_md.write_text("\n".join(lines), encoding="utf-8")
        return

    slowest = sorted(current_map.items(), key=lambda kv: _median(kv[1]), reverse=True)[:10]
    lines.append("## Top Current Benchmarks (median)")
    lines.append("")
    lines.append("| Benchmark | Median | Rounds |")
    lines.append("|---|---:|---:|")
    for name, item in slowest:
        median = _median(item)
        rounds = int(item.get("stats", {}).get("rounds", 0))
        lines.append(f"| `{name}` | {_format_seconds(median)} | {rounds} |")
    lines.append("")

    if baseline_map:
        regressions: list[tuple[str, float, float, float]] = []
        improvements: list[tuple[str, float, float, float]] = []
        for name, current_item in current_map.items():
            baseline_item = baseline_map.get(name)
            if baseline_item is None:
                continue
            cur = _median(current_item)
            base = _median(baseline_item)
            if base <= 0:
                continue
            ratio = cur / base
            if ratio >= regression_threshold:
                regressions.append((name, cur, base, ratio))
            elif ratio <= 1 / regression_threshold:
                improvements.append((name, cur, base, ratio))

        regressions.sort(key=lambda t: t[3], reverse=True)
        improvements.sort(key=lambda t: t[3])

        lines.append(f"## Regressions (>{regression_threshold:.2f}x slower)")
        lines.append("")
        if regressions:
            lines.append("| Benchmark | Current | Baseline | Ratio |")
            lines.append("|---|---:|---:|---:|")
            for name, cur, base, ratio in regressions:
                lines.append(
                    f"| `{name}` | {_format_seconds(cur)} | {_format_seconds(base)} | {ratio:.2f}x |"
                )
        else:
            lines.append("No regressions above threshold.")
        lines.append("")

        lines.append(f"## Improvements (>{regression_threshold:.2f}x faster)")
        lines.append("")
        if improvements:
            lines.append("| Benchmark | Current | Baseline | Ratio |")
            lines.append("|---|---:|---:|---:|")
            for name, cur, base, ratio in improvements:
                lines.append(
                    f"| `{name}` | {_format_seconds(cur)} | {_format_seconds(base)} | {ratio:.2f}x |"
                )
        else:
            lines.append("No improvements above threshold.")
        lines.append("")
    else:
        lines.append("## Baseline Comparison")
        lines.append("")
        lines.append("Baseline JSON not found; current benchmark report generated without comparison.")
        lines.append("")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Current pytest-benchmark JSON")
    parser.add_argument("--output", required=True, type=Path, help="Markdown output path")
    parser.add_argument("--baseline", type=Path, default=None, help="Optional baseline JSON path")
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=1.25,
        help="Ratio threshold for regression/improvement sections",
    )
    args = parser.parse_args()
    generate_report(args.input, args.output, args.baseline, args.regression_threshold)


if __name__ == "__main__":
    main()
