"""How-to: audit one common supplied-return sample without a legacy dependency."""

import argparse
import json
from importlib.metadata import version

import numpy as np
import pandas as pd

from ml4t.diagnostic.compat.alphalens import utils as al
from ml4t.diagnostic.evaluation import PortfolioAnalysis
from ml4t.diagnostic.metrics import compute_forward_returns
from ml4t.diagnostic.signal import prepare_signal_panel


def subday_analysis(*, plots: bool) -> dict:
    dates = pd.date_range("2025-07-07T14:30Z", periods=4, freq="30min")
    index = pd.MultiIndex.from_product([dates, list("ABCD")], names=["date", "asset"])
    factor = pd.Series(np.tile([-3.0, -1.0, 1.0, 3.0], 4), index=index, name="factor")
    outcomes = pd.DataFrame(
        {"30m": np.tile([-0.02, 0.0, 0.02, 0.04], 4) + np.repeat([0, 0.01, -0.01, 0.02], 4)},
        index=index,
    )
    panel = prepare_signal_panel(factor, outcomes, quantiles=None, bins=[-10, -5, 0, 5], max_loss=0)
    result = panel.analyze(min_assets=4, compute_turnover_flag=True)
    assert result.horizon_labels == ("30m",) and result.quantile_ids == (2, 3)
    assert result.turnover == {}  # No daily lag inferred from the supplied subday selector.
    np.testing.assert_allclose(list(result.quantile_returns["30m"].values()), [-0.005, 0.035])
    plotted = None
    if plots:
        from ml4t.diagnostic.visualization.signal import plot_quantile_returns_bar

        figure = plot_quantile_returns_bar(result.to_quantile_result(), period="30m")
        assert list(figure.data[0].x) == ["Q2", "Q3"]
        np.testing.assert_allclose(figure.data[0].y, [-0.005, 0.035])
        plotted = {"labels": list(figure.data[0].x), "means": list(figure.data[0].y)}
    return {
        "supplied_horizons": result.horizon_labels,
        "bucket_ids": result.quantile_ids,
        "means": result.quantile_returns,
        "spread": result.spread,
        "automatic_turnover_lags": result.preparation_metadata["automatic_turnover_lags"],
        "plotted": plotted,
        "limit": "Supplied selectors; return endpoints/calendar not validated by ingestion",
    }


def main(*, plots: bool = False, engineer_outcomes: bool = False) -> None:
    dates = pd.bdate_range("2025-01-02", periods=6, tz="UTC")
    assets = ["A", "B", "C", "D"]
    prices = pd.DataFrame(
        {
            asset: [100.0 + 11 * i + 3 * day + ((day * (i + 2)) % 7) for day in range(6)]
            for i, asset in enumerate(assets)
        },
        index=dates,
    )
    factor = prices.pct_change(fill_method=None).stack(future_stack=True).rename("factor")
    factor.index = factor.index.set_names(["date", "asset"])
    marks = (
        prices.stack(future_stack=True)
        .rename("close")
        .rename_axis(index=["date", "asset"])
        .reset_index()
    )
    forward = compute_forward_returns(marks, periods=[1, 2], group_col="asset")
    forward = forward.set_index(["date", "asset"])[["fwd_ret_1", "fwd_ret_2"]].rename(
        columns={"fwd_ret_1": "1D", "fwd_ret_2": "2D"}
    )
    shifted_returns = al.backshift_returns_series(factor, N=1)
    np.testing.assert_allclose(
        shifted_returns,
        forward["1D"].reindex(shifted_returns.index),
        rtol=1e-12,
        atol=1e-14,
    )
    legacy_clock = al.infer_trading_calendar(factor.index.levels[0], prices.index)
    assert (
        al.add_custom_calendar_timedelta(
            input=dates[0],
            timedelta=pd.Timedelta("1D"),
            freq=legacy_clock,
        )
        == dates[1]
    )
    panel = prepare_signal_panel(
        factor,
        forward,
        quantiles=2,
        groups={"A": "first", "B": "first", "C": "second", "D": "second"},
        max_loss=0.5,
    )
    price_panel = prepare_signal_panel(
        factor,
        prices=marks,
        horizons={"1D": 1, "2D": 2},
        price_basis="raw",
        calendar=None,
        groups={"A": "first", "B": "first", "C": "second", "D": "second"},
        quantiles=2,
        max_loss=0.5,
    )
    assert price_panel.data.equals(panel.data)
    assert price_panel.outcome_data["window_unit"].unique().to_list() == ["observation"]
    adapted = al.get_clean_factor_and_forward_returns(
        factor,
        prices,
        periods=(1, 2),
        quantiles=2,
        filter_zscore=None,
        groupby={"A": "first", "B": "first", "C": "second", "D": "second"},
        max_loss=0.5,
        price_basis="raw",
        calendar=None,
    )
    assert adapted.attrs["diagnostic"]["loss_by_stage"] == panel.metadata["loss_by_stage"]
    adapted_native = adapted.rename(columns={"factor_quantile": "quantile"}).reset_index()
    native = panel.data.to_pandas()
    pd.testing.assert_frame_equal(
        adapted_native[native.columns],
        native,
        check_dtype=False,
        check_categorical=False,
    )
    assert panel.data.height == 12
    bucket_sample = panel.data.to_pandas().set_index(["date", "asset"])
    direct_buckets = al.quantize_factor(bucket_sample, quantiles=2)
    np.testing.assert_array_equal(
        direct_buckets.reindex(bucket_sample.index), bucket_sample["quantile"]
    )
    assert panel.metadata["loss_by_stage"] == {
        "factor": 4,
        "returns": 8,
        "groups": 0,
        "bucketing": 0,
    }
    result = panel.analyze(min_assets=2, compute_turnover_flag=True, autocorrelation_lags=2)
    producer_evidence = None
    if engineer_outcomes:
        import polars as pl
        from ml4t.engineer.labeling import forward_returns
        from ml4t.specs import FeedSpec

        feed = FeedSpec(
            timestamp_col="date", entity_col="asset", time_contract_version=2, calendar=None
        )
        outcomes = forward_returns(
            pl.from_pandas(marks).lazy(),
            horizons={"1D": 1, "2D": 2},
            price_basis="raw",
            entries=pl.from_pandas(factor.reset_index()).select("date", "asset"),
            feed_spec=feed,
            missing_price_policy="null",
        ).collect()
        produced = prepare_signal_panel(
            factor,
            outcomes,
            horizon_col="horizon",
            feed_spec=feed,
            quantiles=2,
            max_loss=0.5,
        )
        actual = produced.data.sort("date", "asset")
        expected = panel.data.sort("date", "asset")
        assert actual.select(pl.col("date").dt.cast_time_unit("ns"), "asset").equals(
            expected.select(pl.col("date").dt.cast_time_unit("ns"), "asset")
        )
        np.testing.assert_array_equal(actual["quantile"], expected["quantile"])
        np.testing.assert_allclose(
            actual.select("1D", "2D").to_numpy(),
            expected.select("1D", "2D").to_numpy(),
            rtol=1e-12,
            atol=1e-12,
        )
        assert produced.analyze(min_assets=2).outcome_data.equals(produced.outcome_data)
        producer_evidence = {
            "engineer_version": version("ml4t-engineer"),
            "outcomes": json.loads(produced.outcome_data.write_json()),
            "same_retained_rows": produced.data.height,
            "endpoint_unit": "observation; labels are not elapsed durations",
        }
    assert result.n_dates == 3 and result.n_assets == 4
    assert sum(result.count_by_quantile.values()) == 12
    ic_component = result.to_ic_result()
    for period, values in ic_component.ic_by_date.items():
        assert abs(ic_component.ic_mean[period] - np.mean(values)) < 1e-14
    grouped_ic = panel.ic_by_date(group_adjust=True, by_group=True, min_obs=2)
    assert grouped_ic.height == 12
    assert grouped_ic["n_obs"].to_list() == [2] * 12
    adjusted = panel.demean_returns(by_group=True)
    unchanged = [name for name in panel.data.columns if name not in ["1D", "2D"]]
    assert adjusted.select(unchanged).equals(panel.data.select(unchanged))
    weights = panel.factor_weights(group_adjust=True)
    gross = weights.to_pandas().groupby("date")["weight"].agg(lambda values: values.abs().sum())
    np.testing.assert_allclose(gross, [1.0] * 3, atol=1e-14, rtol=0)
    portfolio_returns = panel.factor_returns(group_adjust=True)
    factor_regression = panel.factor_alpha_beta(
        annualization_factors={"1D": 252, "2D": 126},
        group_adjust=True,
    )
    assert factor_regression.statistics["n_obs"].to_list() == [3, 3]
    assert factor_regression.observations.height == 6
    for row in factor_regression.statistics.iter_rows(named=True):
        assert row["annual_alpha"] == np.expm1(
            np.log1p(row["alpha_per_period"]) * row["annualization_factor"]
        )
    one_period = portfolio_returns.to_pandas()
    one_period = one_period[one_period["return_column"] == "1D"].set_index("date")[
        "weighted_return"
    ]
    portfolio_summary = PortfolioAnalysis(one_period, periods_per_year=252).compute_summary_stats()
    assert abs(portfolio_summary.total_return - ((1 + one_period).prod() - 1)) < 1e-14
    factor_wealth = panel.factor_cumulative_returns(return_column="1D", group_adjust=True)
    assert factor_wealth.height == 3
    assert abs(factor_wealth["wealth"][-1] - (1 + portfolio_summary.total_return)) < 1e-14
    daily_quantiles = panel.quantile_statistics(by_date=True, demeaned=False)
    quantile_result = result.to_quantile_result()
    empirical = quantile_result.get_dataframe("observations")
    assert empirical.equals(daily_quantiles)
    assert result.quantile_observations.equals(daily_quantiles)
    assert quantile_result.get_monotonicity("1D") == (None, "unavailable")
    assert quantile_result.get_dataframe("spread")["is_monotonic"].to_list() == [None, None]
    equal_date_quantiles = panel.quantile_statistics(demeaned=False)
    assert daily_quantiles.height == 12 and equal_date_quantiles.height == 4
    assert equal_date_quantiles["n_obs"].to_list() == [3] * 4
    spreads = panel.quantile_spread(
        upper_quantile=2, lower_quantile=1, by_date=True, demeaned=False
    )
    assert spreads.height == 6
    assert spreads["upper_n_obs"].to_list() == spreads["lower_n_obs"].to_list() == [2] * 6
    dated_turnover = panel.quantile_turnover(lags=(1, 2))
    assert dated_turnover.height == 12
    assert set(dated_turnover["lag_unit"].to_list()) == {"observations"}
    dated_rank = panel.rank_autocorrelation(lags=(1, 2))
    assert dated_rank.height == 6
    assert set(dated_rank["rank_scope"].to_list()) == {"date"}
    assert set(dated_rank["correlation_method"].to_list()) == {"pearson"}
    persistence = result.to_turnover_result()
    assert persistence is not None
    assert persistence.get_dataframe("dated_turnover").equals(dated_turnover)
    assert persistence.get_dataframe("dated_autocorrelation").equals(dated_rank)
    assert result.to_tear_sheet().turnover_analysis is not None
    spread_plot = None
    persistence_plots = None
    monthly_ic_plot = None
    if plots:
        from ml4t.diagnostic.visualization.signal import (
            plot_autocorrelation,
            plot_monthly_ic_heatmap,
            plot_spread_timeseries,
            plot_top_bottom_turnover,
        )

        figure = plot_spread_timeseries(result.to_quantile_result(), rolling_window=2)
        spread_plot = {
            "metadata": dict(figure.layout.meta),
            "traces": [{"name": trace.name, "values": list(trace.y)} for trace in figure.data],
        }
        persistence_plots = {}
        for name, function in [
            ("membership", plot_top_bottom_turnover),
            ("rank", plot_autocorrelation),
        ]:
            dated_figure = function(persistence, lag=1)
            assert dated_figure.layout.meta["lag_unit"] == "observations"
            assert len(dated_figure.data[0].x) == result.n_dates
            persistence_plots[name] = {
                "metadata": dict(dated_figure.layout.meta),
                "traces": [
                    {
                        "name": trace.name,
                        "dates": list(trace.x),
                        "values": list(trace.y),
                        "counts_and_status": list(trace.customdata),
                    }
                    for trace in dated_figure.data
                ],
            }
        monthly = plot_monthly_ic_heatmap(ic_component, period="1D")
        assert list(monthly.data[0].x) == ["Jan"]
        np.testing.assert_allclose(
            monthly.data[0].z, [[np.mean(ic_component.ic_by_date["1D"])]], rtol=0, atol=1e-14
        )
        monthly_ic_plot = {
            "months": list(monthly.data[0].x),
            "years": list(monthly.data[0].y),
            "values": np.asarray(monthly.data[0].z).tolist(),
        }
    print(
        json.dumps(
            {
                "versions": {
                    name: version(name) for name in ["ml4t-diagnostic", "pandas", "numpy", "polars"]
                },
                "input_rows": panel.metadata["input_rows"],
                "subday_sparse_analysis": subday_analysis(plots=plots),
                "optional_native_outcome_producer": producer_evidence,
                "native_price_preparation": price_panel.metadata["price_preparation"],
                "legacy_return_alignment": {
                    "shifted_rows": len(shifted_returns),
                    "clock": "common observed date levels",
                    "first_explicit_offset": str(dates[1]),
                    "direct_buckets": len(direct_buckets),
                },
                "retained_rows": panel.metadata["retained_rows"],
                "loss_by_stage": panel.metadata["loss_by_stage"],
                "ic": result.ic,
                "aligned_ic_report_summary": ic_component.get_dataframe("summary").to_dicts(),
                "group_adjusted_ic_by_date": grouped_ic.to_dicts(),
                "group_adjusted_returns": adjusted.to_dicts(),
                "factor_weights": weights.to_dicts(),
                "factor_returns": portfolio_returns.to_dicts(),
                "factor_regression": factor_regression.statistics.to_dicts(),
                "factor_regression_observations": factor_regression.observations.to_dicts(),
                "factor_wealth": factor_wealth.to_dicts(),
                "arithmetic_series_total_return": portfolio_summary.total_return,
                "daily_quantiles": daily_quantiles.to_dicts(),
                "monotonicity": {
                    period: quantile_result.get_monotonicity(period)
                    for period in quantile_result.periods
                },
                "equal_date_quantiles": equal_date_quantiles.to_dicts(),
                "quantile_spreads": spreads.to_dicts(),
                "dated_quantile_turnover": dated_turnover.to_dicts(),
                "dated_rank_autocorrelation": dated_rank.to_dicts(),
                "persistence_summary": persistence.get_dataframe("summary").to_dicts(),
                "spread_plot": spread_plot,
                "persistence_plots": persistence_plots,
                "monthly_ic_plot": monthly_ic_plot,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Inspect actual spread, monthly IC and persistence data (viz extra)",
    )
    parser.add_argument(
        "--engineer-outcomes",
        action="store_true",
        help="Also exercise optional Engineer native outcomes; requires a version exposing labeling.forward_returns",
    )
    arguments = parser.parse_args()
    main(plots=arguments.plots, engineer_outcomes=arguments.engineer_outcomes)
