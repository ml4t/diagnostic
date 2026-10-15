"""Compare published legacy and Diagnostic APIs on shared synthetic inputs.

Run from the repository root with the pinned-package command in the migration guide.
"""

from __future__ import annotations

import argparse
from importlib.metadata import version

import numpy as np
import pandas as pd

from ml4t.diagnostic import analyze_signal
from ml4t.diagnostic.evaluation import PortfolioAnalysis


def returns_input() -> tuple[pd.Series, pd.Series]:
    dates = pd.bdate_range("2025-01-02", periods=252)
    rng = np.random.default_rng(42)
    benchmark = pd.Series(rng.normal(0.0003, 0.007, len(dates)), index=dates)
    strategy = pd.Series(
        0.0002 + 0.8 * benchmark.to_numpy() + rng.normal(0, 0.008, len(dates)),
        index=dates,
    )
    return strategy, benchmark


def compare_portfolios(strategy: pd.Series, benchmark: pd.Series) -> None:
    import pyfolio.timeseries as pyfolio_timeseries

    legacy = pyfolio_timeseries.perf_stats(strategy, factor_returns=benchmark)
    current = PortfolioAnalysis(
        returns=strategy.to_numpy(),
        benchmark=benchmark.to_numpy(),
        dates=strategy.index.to_numpy(),
        risk_free=0.0,
        periods_per_year=252,
    ).compute_summary_stats()
    print("Pyfolio / Diagnostic daily portfolio summary")
    for label, attribute in (
        ("Annual return", "annual_return"),
        ("Annual volatility", "annual_volatility"),
        ("Sharpe ratio", "sharpe_ratio"),
        ("Max drawdown", "max_drawdown"),
        ("Alpha", "alpha"),
        ("Beta", "beta"),
    ):
        print(f"{label}: {legacy[label]:.6f} / {getattr(current, attribute):.6f}")
    np.testing.assert_allclose(legacy["Sharpe ratio"], current.sharpe_ratio)
    np.testing.assert_allclose(legacy["Max drawdown"], current.max_drawdown)


def signal_input() -> tuple[pd.Series, pd.DataFrame]:
    dates = pd.bdate_range("2025-01-02", periods=60)
    assets = [f"asset_{index:02d}" for index in range(12)]
    rng = np.random.default_rng(7)
    scores = rng.normal(size=(50, len(assets)))
    factor_index = pd.MultiIndex.from_product([dates[:50], assets], names=["date", "asset"])
    factor = pd.Series(scores.ravel(), index=factor_index, name="factor")
    prices = np.full((len(dates), len(assets)), 100.0)
    for day in range(1, len(dates)):
        lagged_scores = scores[day - 1] if day - 1 < len(scores) else 0.0
        prices[day] = prices[day - 1] * (
            1 + 0.002 * lagged_scores + rng.normal(scale=0.005, size=len(assets))
        )
    return factor, pd.DataFrame(prices, index=dates, columns=assets)


def current_signal(factor: pd.Series, prices: pd.DataFrame):
    factor_long = factor.reset_index()
    price_long = (
        prices.rename_axis("date")
        .stack()
        .rename("price")
        .rename_axis(index=["date", "asset"])
        .reset_index()
    )
    return analyze_signal(
        factor_long,
        price_long,
        periods=(1,),
        quantiles=5,
        filter_zscore=None,
        min_assets=10,
    )


def compare_signals(factor: pd.Series, prices: pd.DataFrame) -> None:
    import alphalens.performance as alphalens_performance
    import alphalens.utils as alphalens_utils

    cleaned = alphalens_utils.get_clean_factor_and_forward_returns(
        factor,
        prices,
        quantiles=5,
        periods=(1,),
        filter_zscore=None,
        max_loss=0.35,
    )
    legacy_ic = alphalens_performance.factor_information_coefficient(cleaned)["1D"].mean()
    current = current_signal(factor, prices)
    print("Alphalens / Diagnostic one-day signal analysis")
    print(f"Input factor rows: {len(factor)}")
    print(f"Cleaned Alphalens rows: {len(cleaned)}")
    print(f"Mean Spearman IC: {legacy_ic:.6f} / {current.ic['1D']:.6f}")
    print(f"Diagnostic spread: {current.spread['1D']:.6f}")
    np.testing.assert_allclose(legacy_ic, current.ic["1D"])


def show_diagnostic_only() -> None:
    strategy, _ = returns_input()
    summary = PortfolioAnalysis(
        returns=strategy.to_numpy(),
        dates=strategy.index.to_numpy(),
        periods_per_year=252,
    ).compute_summary_stats()
    signal = current_signal(*signal_input())
    assert np.isfinite(summary.sharpe_ratio)
    assert np.isfinite(signal.ic["1D"])
    print(f"Diagnostic Sharpe ratio: {summary.sharpe_ratio:.6f}")
    print(f"Diagnostic one-day signal IC: {signal.ic['1D']:.6f}")


def compare_metrics(strategy: pd.Series) -> None:
    import empyrical

    metrics = (
        ("annual_return", "annual_return", lambda v: empyrical.annual_return(v, annualization=252)),
        (
            "annual_volatility",
            "annual_volatility",
            lambda v: empyrical.annual_volatility(v, annualization=252),
        ),
        ("sharpe_ratio", "sharpe_ratio", lambda v: empyrical.sharpe_ratio(v, annualization=252)),
        ("sortino_ratio", "sortino_ratio", lambda v: empyrical.sortino_ratio(v, annualization=252)),
        ("max_drawdown", "max_drawdown", empyrical.max_drawdown),
        ("calmar_ratio", "calmar_ratio", lambda v: empyrical.calmar_ratio(v, annualization=252)),
        ("omega_ratio", "omega_ratio", empyrical.omega_ratio),
        ("tail_ratio", "tail_ratio", empyrical.tail_ratio),
        ("value_at_risk", "var_95", empyrical.value_at_risk),
        ("conditional_value_at_risk", "cvar_95", empyrical.conditional_value_at_risk),
    )
    print("Empyrical / Diagnostic daily metrics")
    for label, values in (("complete", strategy), ("one NaN", strategy.copy())):
        if label == "one NaN":
            values.iloc[10] = np.nan
        current = PortfolioAnalysis(
            returns=values.to_numpy(),
            dates=values.index.to_numpy(),
            periods_per_year=252,
        ).compute_summary_stats()
        print(f"Input: {label}, rows={len(values)}, valid={values.notna().sum()}")
        for name, current_name, legacy_function in metrics:
            old = float(legacy_function(values))
            new = float(getattr(current, current_name))
            print(f"{name}: {old:.6f} / {new:.6f}")
            if label == "complete":
                np.testing.assert_allclose(old, new)
        old_stability = empyrical.stability_of_timeseries(values)
        print(f"stability_of_timeseries: {old_stability:.6f} / {current.stability:.6f}")

    monthly = pd.Series(
        [0.01, -0.02, 0.015, 0.004, -0.006, 0.012, 0.003, -0.005, 0.011, 0.008, -0.004, 0.009],
        index=pd.date_range("2025-01-01", periods=12, freq="MS"),
    )
    legacy_monthly = empyrical.annual_return(monthly, period="monthly")
    current_monthly = (
        PortfolioAnalysis(
            returns=monthly.to_numpy(), dates=monthly.index.to_numpy(), periods_per_year=12
        )
        .compute_summary_stats()
        .annual_return
    )
    print(f"Monthly annual return: {legacy_monthly:.6f} / {current_monthly:.6f}")
    np.testing.assert_allclose(legacy_monthly, current_monthly)

    annual_risk_free = 0.02
    periodic_risk_free = (1 + annual_risk_free) ** (1 / 252) - 1
    legacy_sharpe = empyrical.sharpe_ratio(
        strategy, risk_free=periodic_risk_free, annualization=252
    )
    current_sharpe = (
        PortfolioAnalysis(
            returns=strategy.to_numpy(), risk_free=annual_risk_free, periods_per_year=252
        )
        .compute_summary_stats()
        .sharpe_ratio
    )
    print(f"Sharpe with 2% annual risk-free: {legacy_sharpe:.6f} / {current_sharpe:.6f}")
    np.testing.assert_allclose(legacy_sharpe, current_sharpe)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("diagnostic", "all", "pyfolio", "alphalens", "empyrical"),
        default="diagnostic",
    )
    task = parser.parse_args().task
    packages = ["ml4t-diagnostic", "numpy", "pandas", "scipy", "polars"]
    if task in ("all", "empyrical"):
        packages.append("empyrical-reloaded")
    if task in ("all", "pyfolio"):
        packages.append("pyfolio-reloaded")
    if task in ("all", "alphalens"):
        packages.append("alphalens-reloaded")
    for package in packages:
        print(f"{package}=={version(package)}")
    if task == "diagnostic":
        show_diagnostic_only()
        return
    if task in ("all", "pyfolio", "empyrical"):
        strategy, benchmark = returns_input()
        if task in ("all", "pyfolio"):
            compare_portfolios(strategy, benchmark)
        if task in ("all", "empyrical"):
            compare_metrics(strategy)
    if task in ("all", "alphalens"):
        compare_signals(*signal_input())


if __name__ == "__main__":
    main()
