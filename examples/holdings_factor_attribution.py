"""Run supplied-weight attribution without a provider, broker or fitted regression."""

from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from ml4t.diagnostic.compat import empyrical
from ml4t.diagnostic.evaluation.factor import compute_holdings_attribution


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", action="store_true", help="Also call optional Empyrical")
    parser.add_argument("--compare-pyfolio", action="store_true", help="Also call optional Pyfolio")
    parser.add_argument("--report", type=Path, help="Write the existing native HTML report")
    parser.add_argument(
        "--pyfolio-report",
        type=Path,
        help="Write the familiar attribution sheet and a separate specific-return histogram",
    )
    args = parser.parse_args()
    dates = pd.DatetimeIndex(["2025-07-07T20:00Z", "2025-07-08T20:00Z"])
    starts = pd.DatetimeIndex(["2025-07-03T20:00Z", "2025-07-07T20:00Z"])
    prices = pd.Series([100.0, 103.0, 101.97], index=starts[:1].append(dates))
    returns = prices.pct_change(fill_method=None).iloc[1:]
    keys = {"timestamp": dates.repeat(2), "asset": ["A", "B", "A", "B"]}
    weights = pl.DataFrame({**keys, "known_at": starts.repeat(2), "weight": [0.5, 0.5, 0.25, 0.75]})
    loadings = pl.DataFrame(
        {
            **keys,
            "known_at": starts.repeat(2),
            "market": [2.0, 1.0, 1.0, 1.0],
            "style": [0.0, 0.5, 2.0, 0.0],
        }
    )
    factors = pl.DataFrame({"timestamp": dates, "market": [0.01, -0.02], "style": [0.02, 0.01]})
    result = compute_holdings_attribution(
        pl.DataFrame({"timestamp": dates, "period_start": starts, "return": returns.to_numpy()}),
        weights,
        factors,
        loadings,
        complete_snapshots=True,
    )
    np.testing.assert_allclose(result.exposures.select("market", "style"), [[1.5, 0.25], [1, 0.5]])
    np.testing.assert_allclose(result.contributions["common_returns"], [0.02, -0.015])
    np.testing.assert_allclose(result.cumulative["total_returns"], [0.03, 0.0197], atol=1e-15)
    np.testing.assert_allclose(result.cumulative["common_returns"], [0.02, 0.00455], atol=1e-15)

    index = pd.MultiIndex.from_arrays([dates.repeat(2), keys["asset"]], names=["dt", "ticker"])
    legacy_weights = pd.Series(weights["weight"].to_numpy(), index=index)
    legacy_loadings = pd.DataFrame(
        loadings.select("market", "style").to_numpy(), index=index, columns=["market", "style"]
    )
    legacy_factors = factors.drop("timestamp").to_pandas().set_axis(dates)
    exposures, contributions = empyrical.perf_attrib(
        returns, legacy_weights, legacy_factors, legacy_loadings
    )
    np.testing.assert_allclose(exposures, result.exposures.select("market", "style"))
    np.testing.assert_allclose(contributions["specific_returns"], [0.01, 0.005], atol=1e-15)
    if args.compare:
        import empyrical as reference

        expected = reference.perf_attrib(
            returns.copy(), legacy_weights.copy(), legacy_factors.copy(), legacy_loadings.copy()
        )
        for got, wanted in zip((exposures, contributions), expected, strict=True):
            pd.testing.assert_frame_equal(got, wanted, rtol=1e-12, atol=1e-15)
        print(
            f"Reference empyrical-reloaded={version('empyrical-reloaded')}: actual public tables match"
        )
    if args.compare_pyfolio:
        import importlib

        from ml4t.diagnostic.compat.pyfolio import perf_attrib as current_pyfolio

        reference_pyfolio = importlib.import_module("pyfolio.perf_attrib")
        dollar_positions = legacy_weights.unstack("ticker") * 1000
        dollar_positions["cash"] = 0.0
        arguments = (returns, dollar_positions, legacy_factors, legacy_loadings)
        actual_pyfolio = current_pyfolio.perf_attrib(*arguments)
        expected_pyfolio = reference_pyfolio.perf_attrib(*arguments)
        for got, wanted in zip(actual_pyfolio, expected_pyfolio, strict=True):
            pd.testing.assert_frame_equal(got, wanted, rtol=1e-12, atol=1e-15)
        got_stats = current_pyfolio.create_perf_attrib_stats(actual_pyfolio[1], actual_pyfolio[0])
        wanted_stats = reference_pyfolio.create_perf_attrib_stats(
            expected_pyfolio[1], expected_pyfolio[0]
        )
        pd.testing.assert_series_equal(got_stats[0], wanted_stats[0], rtol=1e-12, atol=1e-15)
        pd.testing.assert_frame_equal(got_stats[1], wanted_stats[1], rtol=1e-12, atol=1e-15)
        print(
            f"Reference pyfolio-reloaded={version('pyfolio-reloaded')}: complete-book tables and statistics match"
        )
    if args.report:
        from ml4t.diagnostic.visualization.backtest import generate_backtest_tearsheet

        generate_backtest_tearsheet(
            returns=returns.to_numpy(),
            metrics={"holdings_attribution": result},
            include_statistical=False,
            periods_per_year=12,
            output_path=args.report,
        )
        print(f"Supplied-weight report: {args.report}")
    if args.pyfolio_report:
        from ml4t.diagnostic.compat.pyfolio import perf_attrib as pf
        from ml4t.diagnostic.compat.pyfolio import tears
        from ml4t.diagnostic.visualization.report_generation import combine_figures_to_html

        positions = legacy_weights.unstack("ticker") * 1000
        positions["cash"] = 0.0
        figures = tears.create_perf_attrib_tear_sheet(
            returns,
            positions,
            legacy_factors,
            legacy_loadings,
            factor_partitions=None,
            return_fig=True,
            periods_per_year=12,
            output_file=args.pyfolio_report,
        )
        assert len(figures) == 3
        np.testing.assert_allclose(figures[0].data[0].y, [0.03, 0.0197], atol=1e-15)
        histogram = pf.plot_alpha_returns(contributions["specific_returns"])
        assert sum(histogram.data[0].y) == 2
        histogram_path = args.pyfolio_report.with_name(
            args.pyfolio_report.stem + "-specific-returns.html"
        )
        combine_figures_to_html(
            [histogram],
            title="Supplied specific-return distribution",
            output_file=histogram_path,
            include_plotlyjs=True,
        )
        print(f"Familiar offline attribution sheet: {args.pyfolio_report}")
        print(f"Specific-return histogram: {histogram_path}")
    packages = ("ml4t-diagnostic", "pandas", "polars", "numpy")
    if args.report or args.pyfolio_report:
        packages += ("plotly",)
    print({name: version(name) for name in packages})
    print("Independent exposures, common/specific contributions and wealth-linked total passed")
    print("Native known-at validation and legacy observed-label alignment are distinct contracts")


if __name__ == "__main__":
    main()
