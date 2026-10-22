"""Run native stress/live/wealth and familiar Pyfolio workflows on shared dated inputs.

Core: uv run python examples/portfolio_regimes.py --task all
Reports: uv run --extra viz python examples/portfolio_regimes.py --task all --output regimes.html --familiar-output stress.html
Reference: add pyfolio-reloaded==0.10.0 and its compatible pandas==2.3.3, then --compare-pyfolio.
These are supplied observations, not a broker backtest or an exchange-session forecast.
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from ml4t.diagnostic.compat import pyfolio as pf
from ml4t.diagnostic.evaluation import PortfolioAnalysis


def main() -> None:
    """Execute the selected public workflows and optional offline reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("native", "familiar", "all"), default="all")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--familiar-output", type=Path)
    parser.add_argument("--compare-pyfolio", action="store_true")
    args = parser.parse_args()
    dates = pd.date_range("2025-01-13", periods=12, tz="UTC")
    returns = pd.Series(
        [0.01, -0.02, 0.005, 0.02, -0.01, 0.03, -0.005, 0.01, -0.01, 0.02, 0.04, -0.03],
        index=dates,
        name="strategy",
    )
    benchmark = returns * 0.5
    periods = {"selected": (dates[1], dates[4]), "unobserved": ("1990", "1990")}
    live_start = dates[8]
    outcome = {
        "versions": {name: version(name) for name in ("ml4t-diagnostic", "pandas", "numpy")},
        "input": "12 actual supplied observation keys; no session resampling or inferred NAV/fills",
        "live_start": str(live_start),
    }
    if args.task in ("native", "all"):
        analysis = PortfolioAnalysis.from_frames(
            returns, benchmark=benchmark, calendar=None, periods_per_year=252
        )
        stress = analysis.compute_stress(periods)
        live = analysis.compute_live_split(live_start=live_start)
        cones = analysis.compute_wealth_cones(
            live_start=live_start,
            horizon=7,
            n_samples=256,
            method="stationary",
            block_size=3,
            seed=42,
            initial_capital=1000,
        )
        assert cones.training.height == 8 and cones.live.height == 4
        assert cones.paths.height == 256 * 7
        assert stress.coverage["n_observations"].to_list() == [4, 0]
        changed = returns.copy()
        changed.loc[live_start:] = [0.5, -0.8, 0.9, -0.7]
        independent = PortfolioAnalysis.from_frames(
            changed, benchmark=benchmark, calendar=None
        ).compute_wealth_cones(
            live_start=live_start,
            horizon=7,
            n_samples=256,
            method="stationary",
            block_size=3,
            seed=42,
            initial_capital=1000,
        )
        assert cones.paths.equals(independent.paths) and cones.bands.equals(independent.bands)
        outcome["native"] = {
            "coverage": stress.coverage.to_dicts(),
            "method": "stationary circular blocks; expected3training observations",
            "n_training": cones.training.height,
            "n_live": cones.live.height,
            "forecast_rows": cones.paths.height,
            "horizon_unit": "observations",
            "post_live_perturbation": "same training draws and bounds",
            "bands": "explicit quantiles and mean +/- population std, not mean confidence intervals",
        }
        if args.output:
            analysis.create_tear_sheet(stress=stress, live=live, wealth_cones=cones).save_html(
                args.output, include_plotlyjs=True
            )
            outcome["native"]["report"] = str(args.output)
    if args.task in ("familiar", "all"):
        training = returns.loc[returns.index < live_start]
        forecast = pf.timeseries.forecast_cone_bootstrap(
            training, 4, num_samples=256, random_seed=42, starting_value=1000
        )
        selected = pf.timeseries.extract_interesting_date_ranges(returns, periods)
        assert list(selected) == ["selected"] and len(selected["selected"]) == 4
        outcome["familiar"] = {
            "shape": list(forecast.shape),
            "columns": forecast.columns.to_list(),
            "event_observations": 4,
            "rng": "local legacy RandomState; not native Generator",
            "definition": "mean +/- population std; actual return samples preserve missingness",
        }
        if args.familiar_output:
            pf.tears.create_interesting_times_tear_sheet(
                returns, benchmark, periods, output_file=args.familiar_output
            )
            outcome["familiar"]["report"] = str(args.familiar_output)
        if args.compare_pyfolio:
            from pyfolio import timeseries as reference

            expected = reference.forecast_cone_bootstrap(
                training, 4, num_samples=256, random_seed=42, starting_value=1000
            )
            pd.testing.assert_frame_equal(
                forecast, expected, check_exact=False, rtol=1e-13, atol=1e-13
            )
            np.testing.assert_array_equal(
                pf.timeseries.simulate_paths(training, 4, num_samples=256, random_seed=42),
                reference.simulate_paths(training, 4, num_samples=256, random_seed=42),
            )
            outcome["familiar"]["reference"] = {
                "version": version("pyfolio-reloaded"),
                "raw_draws_and_all_bounds": "matched on identical samples/options",
            }
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    main()
