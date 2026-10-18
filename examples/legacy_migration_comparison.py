"""Compare published legacy APIs with Diagnostic on shared synthetic inputs.

Run from the repository root with the pinned-package command in the migration guide.
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path

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
    legacy_series = alphalens_performance.factor_information_coefficient(cleaned)["1D"]
    legacy_ic = legacy_series.mean()
    current = current_signal(factor, prices)
    print("Alphalens / Diagnostic one-day signal analysis")
    print(f"Input factor rows: {len(factor)}")
    print(f"Cleaned Alphalens rows: {len(cleaned)}")
    print(f"Mean Spearman IC: {legacy_ic:.6f} / {current.ic['1D']:.6f}")
    print(f"Diagnostic spread: {current.spread['1D']:.6f}")
    np.testing.assert_allclose(legacy_ic, current.ic["1D"])
    try:
        from ml4t.diagnostic import analyze_prepared_signal
    except ImportError:
        print(
            "Installed Diagnostic has no prepared-panel route; the original comparison above ran."
        )
        return
    prepared = analyze_prepared_signal(cleaned, periods=(1,), quantiles=5, min_assets=2)
    canonical = (
        prepared.prepared_data.to_pandas()
        .rename(columns={"quantile": "factor_quantile", "1D_fwd_return": "1D"})
        .set_index(["date", "asset"])
    )
    pd.testing.assert_frame_equal(
        canonical[cleaned.columns].sort_index(), cleaned.sort_index(), check_freq=False
    )
    np.testing.assert_array_equal(
        pd.DatetimeIndex(prepared.ic_dates["1D"]).asi8, legacy_series.index.asi8
    )
    np.testing.assert_allclose(prepared.ic_series["1D"], legacy_series, atol=1e-12)
    legacy_quantiles, _ = alphalens_performance.mean_return_by_quantile(
        cleaned, by_date=False, demeaned=False, group_adjust=False
    )
    for bucket in range(1, 6):
        np.testing.assert_allclose(
            prepared.quantile_returns["1D"][bucket],
            legacy_quantiles.loc[bucket, "1D"],
            atol=1e-12,
        )
    print(f"Prepared rows/buckets retained exactly: {prepared.prepared_data.height}")
    print(
        "Prepared per-date IC and means match on this balanced sample; "
        "native std/spread inference differ."
    )
    compare_factor_regression(factor, prices)


def compare_factor_regression(factor: pd.Series, prices: pd.DataFrame) -> None:
    """Compare actual dated samples before coefficients, using two supplied horizons."""
    import itertools

    import alphalens.performance as reference
    import alphalens.utils as reference_utils

    from ml4t.diagnostic.signal import prepare_signal_panel

    groups = pd.Series(
        [
            "first" if (int(asset[-2:]) + date.day) % 12 < 4 else "second"
            for date, asset in factor.index
        ],
        index=factor.index,
    )
    sample = reference_utils.get_clean_factor_and_forward_returns(
        factor,
        prices,
        groupby=groups,
        quantiles=5,
        periods=(1, 5),
        filter_zscore=None,
        max_loss=0,
    )
    panel = prepare_signal_panel(
        sample["factor"],
        sample[["1D", "5D"]],
        groups=sample["group"],
        quantiles=5,
        max_loss=0,
    )
    canonical = panel.data.to_pandas().set_index(["date", "asset"])
    pd.testing.assert_index_equal(canonical.index, sample.index, exact=False)
    np.testing.assert_allclose(canonical[["1D", "5D"]], sample[["1D", "5D"]], atol=0, rtol=0)
    assert canonical["group"].tolist() == sample["group"].tolist()
    universe = sample[["1D", "5D"]].groupby(level="date").mean()
    for flags in itertools.product([False, True], repeat=3):
        options = dict(zip(["demeaned", "group_adjust", "equal_weight"], flags, strict=True))
        expected_weights = reference.factor_weights(sample, **options)
        weights = panel.factor_weights(**options).to_pandas().set_index(["date", "asset"])
        np.testing.assert_allclose(weights["weight"], expected_weights, atol=1e-14, rtol=1e-12)
        expected_returns = reference.factor_returns(sample, **options)
        expected = reference.factor_alpha_beta(sample, **options)
        result = panel.factor_alpha_beta(annualization_factors={"1D": 252, "5D": 50.4}, **options)
        actual = result.statistics.to_pandas().set_index("return_column")
        for column in ["1D", "5D"]:
            observed = result.observations.to_pandas()
            observed = observed[observed.return_column == column].set_index("date")
            pd.testing.assert_index_equal(observed.index, expected_returns.index, exact=False)
            np.testing.assert_allclose(
                observed.portfolio_return, expected_returns[column], atol=1e-14, rtol=1e-12
            )
            np.testing.assert_allclose(
                observed.universe_return, universe[column], atol=1e-14, rtol=1e-12
            )
            assert observed.paired.all() and observed.n_universe.eq(12).all()
            assert observed.n_portfolio.eq(12).all()
            assert actual.loc[column, "n_obs"] == len(expected_returns)
            np.testing.assert_allclose(
                actual.loc[column, ["annual_alpha", "beta"]].to_numpy(dtype=float),
                expected.loc[["Ann. alpha", "beta"], column].to_numpy(dtype=float),
                atol=1e-10,
                rtol=1e-8,
            )
    print(
        "Factor regression: eight weighting variants, two horizons, exact dated membership/counts"
    )
    supplied = universe.mul([0.6, -0.2]).add([0.001, -0.002]).iloc[::-1]
    supplied.iloc[3, 0] = np.nan
    before = supplied.copy(deep=True)
    expected = reference.factor_alpha_beta(sample, returns=supplied)
    result = panel.factor_alpha_beta(
        returns=supplied, annualization_factors={"1D": 252, "5D": 50.4}
    )
    actual = result.statistics.to_pandas().set_index("return_column")
    np.testing.assert_allclose(actual["beta"], [0.6, -0.2], atol=1e-12, rtol=1e-8)
    np.testing.assert_allclose(
        actual["annual_alpha"], expected.loc["Ann. alpha"], atol=1e-10, rtol=1e-8
    )
    assert actual["n_obs"].tolist() == [len(supplied) - 1, len(supplied)]
    pd.testing.assert_frame_equal(supplied, before)
    print(
        "Provided factor returns: reverse-date alignment and horizon-specific missingness retained"
    )
    flat_sample = reference_utils.get_clean_factor_and_forward_returns(
        factor,
        prices * 0 + 100,
        quantiles=5,
        periods=(1,),
        filter_zscore=None,
        max_loss=0,
    )
    flat_panel = prepare_signal_panel(
        flat_sample["factor"],
        flat_sample[["1D"]],
        quantiles=5,
        max_loss=0,
    )
    legacy_constant = reference.factor_alpha_beta(flat_sample)
    native_constant = flat_panel.factor_alpha_beta(annualization_factors={"1D": 252})
    assert legacy_constant.loc["beta", "1D"] == 0
    assert native_constant.statistics["beta"][0] is None
    assert native_constant.statistics["design_rank"][0] == 1
    assert native_constant.statistics["status"][0] == "constant_benchmark"
    print(
        "Flat-price reconciliation: reference beta0 versus unidentified slope (rank1), not parity"
    )


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
        try:
            current = PortfolioAnalysis(
                returns=values.to_numpy(),
                dates=values.index.to_numpy(),
                periods_per_year=252,
            ).compute_summary_stats()
        except ValueError as exc:
            if label != "one NaN" or "finite" not in str(exc):
                raise
            current = None
            print(f"Diagnostic rejects missing observations: {exc}")
        print(f"Input: {label}, rows={len(values)}, valid={values.notna().sum()}")
        for name, current_name, legacy_function in metrics:
            old = float(legacy_function(values))
            new = float(getattr(current, current_name)) if current is not None else None
            displayed = f"{new:.6f}" if new is not None else "unavailable (strict input boundary)"
            print(f"{name}: {old:.6f} / {displayed}")
            if label == "complete":
                np.testing.assert_allclose(old, new)
        old_stability = empyrical.stability_of_timeseries(values)
        stability = f"{current.stability:.6f}" if current is not None else "unavailable"
        print(f"stability_of_timeseries: {old_stability:.6f} / {stability}")

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


def show_backtest_exposure() -> None:
    from datetime import UTC, datetime

    import polars as pl
    from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy

    from ml4t.diagnostic.integration import analyze_backtest_result, generate_tearsheet_from_result

    class OnceShort(Strategy):
        def __init__(self):
            self.submitted = False

        def on_data(self, timestamp, data, context, broker):
            if "A" in data and not self.submitted:
                broker.submit_order("A", -10)
                self.submitted = True

    for final_price in [150.0, 200.0, 250.0]:
        marks = [100.0, 100.0, final_price]
        prices = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, day, 15, tzinfo=UTC) for day in [2, 3, 6]],
                "asset": ["A"] * 3,
                **dict.fromkeys(["open", "high", "low", "close"], marks),
                "volume": [10000.0] * 3,
            }
        )
        result = Engine(
            DataFeed(prices_df=prices),
            OnceShort(),
            BacktestConfig(initial_cash=1000, allow_short_selling=True),
        ).run()
        assert len(result.fills) == 1
        assert result.fills[0].quantity == 10 and result.fills[0].price == 100
        assert result.fills[0].timestamp == prices["timestamp"][1]
        occupancy = analyze_backtest_result(result).occupancy
        last = occupancy["timeline"].row(-1, named=True)
        assert last["cash"] == 2000
        assert last["equity"] == 2000 - 10 * final_price
        assert last["gross_exposure"] == 10 * final_price
        assert last["net_exposure"] == -10 * final_price
        assert last["open_positions"] == 1
        if final_price == 150:
            assert last["gross_exposure_fraction"] == 3
            assert last["net_exposure_fraction"] == -3
        else:
            assert last["gross_exposure_fraction"] is None
            assert last["net_exposure_fraction"] is None
            assert occupancy["coverage"]["n_normalized_observations"] == 2
        print(f"Final mark {final_price}: {last}")
        print(f"Coverage: {occupancy['coverage']}")
        if final_price == 200:
            html = generate_tearsheet_from_result(result, include_statistical=False)
            assert "2 of 3 snapshots" in html and "defined ratios only" in html
        elif final_price == 250:
            try:
                generate_tearsheet_from_result(result, include_statistical=False)
            except ValueError as exc:
                assert "simple returns below -1" in str(exc)
                print(f"Full performance report remains unsupported: {exc}")
            else:
                raise AssertionError("Expected the explicit simple-return domain restriction")


def show_backtest_activity() -> None:
    from datetime import UTC, datetime

    import polars as pl
    from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy

    from ml4t.diagnostic.integration import analyze_backtest_result, generate_tearsheet_from_result

    class ShortThenCover(Strategy):
        def __init__(self):
            self.observations = 0

        def on_data(self, timestamp, data, context, broker):
            self.observations += 1
            if self.observations == 1:
                broker.submit_order("A", -10)
            elif self.observations == 2:
                broker.submit_order("A", 10)

    for mark in [150.0, 200.0, 250.0]:
        prices = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, day, 15, tzinfo=UTC) for day in [2, 3, 6]],
                "asset": ["A"] * 3,
                **dict.fromkeys(["open", "high", "low", "close"], [100.0, 100.0, mark]),
                "volume": [10000.0] * 3,
            }
        )
        result = Engine(
            DataFeed(prices_df=prices),
            ShortThenCover(),
            BacktestConfig(initial_cash=1000, allow_short_selling=True),
        ).run()
        assert len(result.fills) == 2
        assert [fill.price for fill in result.fills] == [100, mark]
        assert [fill.quantity for fill in result.fills] == [10, 10]
        assert result.fills[-1].timestamp == prices["timestamp"][-1]
        activity = analyze_backtest_result(result).activity
        last = activity["turnover_timeline"].row(-1, named=True)
        assert last["equity"] == 2000 - 10 * mark
        assert last["filled_notional"] == 10 * mark
        assert activity["metrics"]["total_filled_notional"] == 1000 + 10 * mark
        if mark == 150:
            assert last["turnover"] == 3 and activity["metrics"]["avg_turnover"] == 2
        else:
            assert last["turnover"] is None and last["cost_drag"] is None
            assert activity["metrics"]["avg_turnover"] == 1
            assert activity["coverage"]["n_normalized_active_observations"] == 1
        if mark == 200:
            html = generate_tearsheet_from_result(result, include_statistical=False)
            assert "2 of 3 observations" in html and "1 of 2 active observations" in html
        print(f"Closing mark {mark}: {last}")
        print(f"Activity metrics: {activity['metrics']}")
        print(f"Normalization coverage: {activity['coverage']}")


def _prepared_holdings_inputs() -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
    returns = pd.Series([0.0, 0.0, 0.0], index=dates)
    positions = pd.DataFrame({"A": [0.0, 400.0, 400.0], "B": [0.0, -300.0, -300.0]}, index=dates)
    valuations = pd.DataFrame({"equity": [1000.0] * 3, "cash": [1000.0, 900.0, 900.0]}, index=dates)
    return returns, positions, valuations


def show_prepared_holdings() -> None:
    import polars as pl

    returns, positions, valuations = _prepared_holdings_inputs()
    dates = returns.index
    canonical = positions.reset_index(names="date").melt(
        id_vars="date", var_name="asset", value_name="value"
    )
    canonical["asset"] = canonical["asset"].map({"A": 1, "B": 2})
    for shape, snapshot, short_id in [("wide", positions, "B"), ("integer-ID long", canonical, 2)]:
        analysis = PortfolioAnalysis(returns, positions=snapshot)
        long_id = "A" if shape == "wide" else 1
        mapping = {long_id: "Technology", short_id: "Finance"}
        holdings = analysis.compute_holdings_analysis(
            valuations, complete_snapshots=True, sector_mapping=mapping
        )
        assert holdings.sectors is not None
        final = holdings.timeline.row(-1, named=True)
        np.testing.assert_allclose(
            [
                final[name]
                for name in [
                    "long_exposure_fraction",
                    "short_exposure_fraction",
                    "gross_exposure_fraction",
                    "net_exposure_fraction",
                    "largest_gross_share",
                    "gross_hhi",
                ]
            ],
            [0.4, 0.3, 0.7, 0.1, 4 / 7, 25 / 49],
        )
        assert holdings.positions.filter(pl.col("asset") == short_id)["nav_weight"][-1] == -0.3
        print(f"Complete {shape} snapshots; positive short magnitude, canonical IDs A=1/B=2")
        print(holdings.timeline)
        print(holdings.positions)
        print(holdings.coverage)
        assert holdings.top_positions(side="short").rows() == [(short_id, -300.0)]
        assert holdings.top_positions(top=1).rows() == [(long_id, 400.0)]
        weight_columns = [
            "max_long_weight",
            "median_long_weight",
            "median_short_weight",
            "min_short_weight",
        ]
        np.testing.assert_allclose(
            holdings.timeline.select(weight_columns).row(-1), [0.4, 0.4, -0.3, -0.3]
        )
        assert holdings.timeline.select(weight_columns).row(0) == (None,) * 4
        print("Across-sample top marked values, not normalized weights")
        print(holdings.top_positions())
        print(holdings.timeline.select("date", *weight_columns))
        final_sectors = holdings.sectors.filter(pl.col("date") == dates[-1])
        assert final_sectors.filter(pl.col("sector") == "Technology")["net_exposure"][0] == 400
        assert final_sectors.filter(pl.col("sector") == "Finance")["net_exposure"][0] == -300
        print("Asset sectors only; account cash remains in timeline")
        print(holdings.sectors)
        unclassified = analysis.compute_holdings_analysis(
            valuations, complete_snapshots=True, sector_mapping={long_id: "Technology"}
        )
        assert unclassified.sectors is not None
        unknown = unclassified.sectors.filter(
            pl.col("sector").is_null() & (pl.col("date") == dates[-1])
        )
        assert unknown["net_exposure"][0] == -300
        print("Unmapped short remains visible in the null-sector group")
        unavailable = analysis.compute_holdings_analysis(
            valuations.drop(columns="equity"), complete_snapshots=True, sector_mapping=mapping
        )
        assert unavailable.sectors is not None
        assert unavailable.timeline["gross_exposure_fraction"].null_count() == 3
        assert unavailable.timeline["gross_exposure"][-1] == 700
        assert unavailable.sectors["gross_exposure_fraction"].null_count() == 6
        print("Without reported NAV, raw holdings remain available and NAV ratios are null")


def show_prepared_holdings_report(output: Path) -> None:
    from ml4t.diagnostic.visualization.portfolio import create_portfolio_dashboard

    returns, positions, valuations = _prepared_holdings_inputs()
    analysis = PortfolioAnalysis(returns, positions=positions)
    holdings = analysis.compute_holdings_analysis(
        valuations, complete_snapshots=True, sector_mapping={"A": "Technology"}
    )
    report = create_portfolio_dashboard(analysis, holdings=holdings, include_transactions=False)
    exposure = {trace.name: list(trace.y) for trace in report.figures["Holdings Exposure"].data}
    np.testing.assert_allclose(exposure["Gross"], [0, 0.7, 0.7])
    np.testing.assert_allclose(exposure["Short (signed)"], [0, -0.3, -0.3])
    data = report.to_dict()
    assert data["holdings"]["timeline"][-1]["cash"] == 900
    assert data["holdings"]["top_positions"]["short"] == [{"asset": "B", "value": -300.0}]
    report.save_html(output, include_plotlyjs=True)
    print(f"Prepared holdings HTML: {output}")
    print("Embedded Plotly; no Backtest or legacy runtime dependency")
    print("Unmapped short B remains visible; account cash stays separate from asset sectors")
    print(report.availability)
    print(list(report.figures))


def show_empyrical_native() -> None:
    """Run captured-price scalar and path adapters without a legacy dependency."""
    from ml4t.diagnostic.compat import empyrical as metrics

    prices = pd.DataFrame(
        {"complete": [100, 102, 101, 103, 104], "missing": [50, 51, 52, 51, np.nan]},
        index=pd.bdate_range("2025-01-02", periods=5, tz="UTC"),
    )
    returns = prices.pct_change(fill_method=None).iloc[1:]
    before = returns.copy(deep=True)
    annual = metrics.annual_return(returns, annualization=12)
    np.testing.assert_allclose(annual, [1.04**3 - 1, 1.02**3 - 1])
    for name in [
        "annual_return",
        "cagr",
        "annual_volatility",
        "calmar_ratio",
        "sharpe_ratio",
        "sortino_ratio",
        "downside_risk",
    ]:
        kwargs = {"annualization": 12}
        if name == "sharpe_ratio":
            kwargs["risk_free"] = 0.001
        if name in {"sortino_ratio", "downside_risk"}:
            kwargs["required_return"] = 0.001
        function = getattr(metrics, name)
        result = (
            returns.apply(lambda column, call=function, options=kwargs: call(column, **options))
            if name == "calmar_ratio"
            else function(returns, **kwargs)
        )
        print(f"{name}: {type(result).__name__}; {np.asarray(result).tolist()}")
    out = np.empty(2)
    result = metrics.sortino_ratio(returns, required_return=0.001, annualization=12, out=out)
    assert list(result.index) == ["complete", "missing"] and np.shares_memory(result, out)
    assert metrics.annualization_factor("ignored", 52.5) == 52.5
    pd.testing.assert_frame_equal(returns, before)
    print("Empyrical scalar adapters: 7 routes; 4 dated rows; 2 columns; 1 missing return")
    path_prices = pd.DataFrame(
        {"strategy": [100.0, 101.0, 98.0, np.nan, 103.0], "benchmark": [50, 50, 52, 51, 53]},
        index=pd.date_range("2025-01-02", periods=5, tz="UTC", name="date"),
    )
    paths = pd.DataFrame(
        metrics.simple_returns(path_prices.to_numpy()),
        index=path_prices.index[1:],
        columns=path_prices.columns,
    )
    wealth = metrics.cum_returns(paths, starting_value=100)
    np.testing.assert_allclose(wealth, [[101, 100], [98, 104], [98, 102], [98, 106]])
    np.testing.assert_allclose(metrics.cum_returns_final(paths), [-0.02, 0.06], atol=1e-15)
    drawdown = metrics.drawdown_series(paths)
    np.testing.assert_allclose(
        drawdown, [[0, 0], [-3 / 101, 0], [-3 / 101, -1 / 52], [-3 / 101, 0]], atol=1e-15
    )
    np.testing.assert_allclose(metrics.max_drawdown(paths), [-3 / 101, -1 / 52], atol=1e-15)
    assert wealth.index.equals(paths.index) and drawdown.columns.equals(paths.columns)
    print(
        "Empyrical return paths: 5 routes; missing adjacent prices retained; initial capital checked"
    )


def show_aligned_events(output: Path, *, complete_report: bool = False) -> None:
    """Build a grouped event report from one owned, price-produced signal panel."""
    from ml4t.diagnostic.signal import AlignedEventResult, prepare_signal_panel
    from ml4t.diagnostic.visualization.signal import generate_aligned_event_report

    dates = pd.bdate_range("2025-01-02", periods=8, tz="UTC")
    prices = pd.DataFrame(
        np.arange(100.0, 106.0)
        * np.exp(np.arange(8)[:, None] * np.array([0.01, 0.02, -0.01, 0.03, -0.02, 0.005])),
        index=dates,
        columns=list("ABCDEF"),
    )
    factor = prices.iloc[[0, 2, 4, 6]].stack().rename("factor")
    factor.index.names = ["date", "asset"]
    forward = prices.pct_change(fill_method=None).shift(-1).stack().rename("1D").to_frame()
    forward.index.names = ["date", "asset"]
    groups = pd.Series(
        [
            "small" if asset in ("AB" if day < dates[4] else "AC") else "large"
            for day, asset in factor.index
        ],
        index=factor.index,
        name="group",
    )
    panel = prepare_signal_panel(factor, forward, quantiles=2, groups=groups, max_loss=0)
    analyze = panel.event_returns if complete_report else panel.event_study
    result = analyze(
        prices,
        before=1,
        after=2,
        input_kind="levels",
        demeaned=False,
        group_adjust=False,
        by_group=True,
    )
    if complete_report:
        from ml4t.diagnostic.signal import EventReturnsResult
        from ml4t.diagnostic.visualization.signal import generate_event_returns_report

        assert result.overall.paths.height == 96 and result.groups is not None
        assert EventReturnsResult.from_json(result.to_json()).groups.paths.equals(
            result.groups.paths
        )
        path = generate_event_returns_report(result, output_file=output)
        output.with_suffix(".json").write_text(result.to_json(), encoding="utf-8")
        print(f"Complete event-returns HTML: {path}")
        print("5 panels: overall, 2 quantile deviations, 2 group overviews; typed JSON exported")
        return
    assert panel.data.height == 24 and result.paths.height == 96
    assert result.event_counts["n_events"].sum() == 24
    assert AlignedEventResult.from_json(result.to_json()).paths.equals(result.paths)
    path = generate_aligned_event_report(result, output_file=output)
    assert Path(path).is_file()
    print(f"Aligned event HTML: {path}")
    print("24 declared events; 96 requested path cells; dynamic groups and buckets retained")
    print("Supplied levels, sample standard deviations and observed-row offsets; not CAR")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=(
            "diagnostic",
            "all",
            "pyfolio",
            "alphalens",
            "empyrical",
            "empyrical-native",
            "exposure",
            "activity",
            "holdings",
            "holdings-report",
            "events",
            "event-returns",
        ),
        default="diagnostic",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="HTML path for --task holdings-report, events or event-returns",
    )
    arguments = parser.parse_args()
    task = arguments.task
    packages = ["ml4t-diagnostic", "numpy", "pandas", "scipy", "polars"]
    if task in ("exposure", "activity"):
        packages.extend(["ml4t-backtest", "plotly"])
    if task in ("holdings-report", "events", "event-returns"):
        packages.append("plotly")
    if task in ("all", "empyrical"):
        packages.append("empyrical-reloaded")
    if task in ("all", "pyfolio"):
        packages.append("pyfolio-reloaded")
    if task in ("all", "alphalens"):
        packages.append("alphalens-reloaded")
    for package in packages:
        print(f"{package}=={version(package)}")
    if task == "empyrical-native":
        show_empyrical_native()
        return
    if task == "holdings-report":
        show_prepared_holdings_report(arguments.output or Path("holdings-report.html"))
        return
    if task == "events":
        show_aligned_events(arguments.output or Path("aligned-events.html"))
        return
    if task == "event-returns":
        show_aligned_events(arguments.output or Path("event-returns.html"), complete_report=True)
        return
    if task in ("all", "holdings"):
        show_prepared_holdings()
        if task == "holdings":
            return
    if task == "exposure":
        show_backtest_exposure()
        return
    if task == "activity":
        show_backtest_activity()
        return
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
