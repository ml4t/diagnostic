"""Requested persistence lags must survive whole public report workflows."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.results.signal_results import TurnoverAnalysisResult
from ml4t.diagnostic.signal import prepare_signal_panel


def captured_turnover(profile="rotating", *, lags=(1, 2, 5), nanoseconds=False):
    dates = pd.bdate_range("2025-03-03", periods=14, tz="UTC")
    if nanoseconds:
        dates = pd.DatetimeIndex([dates[0] + pd.Timedelta(i * 100, "ns") for i in range(14)])
    rates = (
        [0.03, 0.02, 0.01, -0.01, -0.02, -0.03]
        if profile == "three_buckets"
        else [0.01, 0.02, -0.01, -0.02]
    )
    increments = np.array([np.roll(rates, i) for i in range(13)])
    if profile in {"constant", "singleton"}:
        increments[:] = 0.01
    prices = pd.DataFrame(
        np.vstack([np.full(len(rates), 100.0), 100 * np.cumprod(1 + increments, axis=0)]),
        index=dates,
        columns=list("ABCDEF")[: len(rates)],
    )
    if profile == "singleton":
        prices = prices[["A"]]
    factor = (
        prices.pct_change(fill_method=None)
        .iloc[1:-1]
        .stack(future_stack=True)
        .rename("factor")
        .rename_axis(["date", "asset"])
    )
    if profile == "departures":
        factor = factor.drop((dates[4], "C"))
    if profile == "date_gap":
        factor = factor.drop(dates[6], level="date")
    forward = (
        (prices.shift(-1) / prices - 1)
        .stack(future_stack=True)
        .rename("1D")
        .rename_axis(["date", "asset"])
        .to_frame()
    )
    options = (
        {"quantiles": None, "bins": 1}
        if profile in {"constant", "singleton"}
        else {"quantiles": 3 if profile == "three_buckets" else 2}
    )
    panel = prepare_signal_panel(factor, forward, max_loss=0, **options)
    component = TurnoverAnalysisResult.from_observations(
        panel.quantile_turnover(lags=lags), panel.rank_autocorrelation(lags=lags)
    )
    return panel, component, dates[1:-1]


def _key(figure):
    metadata = figure.layout.meta
    return ("turnover" if "turnover_method" in metadata else "rank", metadata["lag"])


def test_whole_plot_collection_does_not_omit_computed_custom_lags_or_change_caller_data():
    from ml4t.diagnostic.visualization.signal import plot_signal_turnover

    panel, component, _ = captured_turnover(lags=(5, 1, 2))
    before, data = component.model_dump_json(), panel.data
    random_state = np.random.get_state()
    figures = plot_signal_turnover(component)
    assert [_key(figure) for figure in figures] == [
        (kind, lag) for kind in ["turnover", "rank"] for lag in [5, 1, 2]
    ]
    for figure in figures:
        kind, lag = _key(figure)
        assert all(len(trace.y) == 12 for trace in figure.data)
        assert all(all(value is None for value in trace.y[:lag]) for trace in figure.data)
        if kind == "turnover":
            expected = 1 if lag == 2 else 0.5
            assert len(figure.data) == 2
            for trace in figure.data:
                np.testing.assert_allclose(trace.y[lag:], expected, atol=1e-12)
        else:
            expected = -1 if lag == 2 else 0
            np.testing.assert_allclose(figure.data[0].y[lag:], expected, atol=1e-12)
        assert figure.layout.meta["lag_unit"] == "observations"
    assert component.model_dump_json() == before and panel.data.equals(data)
    after = np.random.get_state()
    assert after[0] == random_state[0] and after[2:] == random_state[2:]
    np.testing.assert_array_equal(after[1], random_state[1])


def test_all_quantile_option_exposes_the_actual_middle_bucket_without_changing_result():
    from ml4t.diagnostic.visualization.signal import plot_signal_turnover

    _, component, _ = captured_turnover("three_buckets")
    before = component.model_dump_json()
    extremes = plot_signal_turnover(component)
    all_buckets = plot_signal_turnover(component, show_all_quantiles=True)
    assert [trace.name for trace in extremes[0].data] == ["Q3", "Q1"]
    assert [trace.name for trace in all_buckets[0].data] == ["Q1", "Q2", "Q3"]
    np.testing.assert_allclose(all_buckets[0].data[1].y[1:], 0.5, atol=1e-12)
    assert component.model_dump_json() == before


def test_report_lags_preserve_supplied_empty_observations_instead_of_compressing_the_clock():
    from ml4t.diagnostic.visualization.signal import plot_signal_turnover

    panel, observed, dates = captured_turnover("date_gap", lags=(1, 2))
    supplied = TurnoverAnalysisResult.from_observations(
        panel.quantile_turnover(lags=(1, 2), date_grid=pd.DataFrame({"date": dates})),
        panel.rank_autocorrelation(lags=(1, 2), date_grid=pd.DataFrame({"date": dates})),
    )
    compact = plot_signal_turnover(observed)[0]
    full = plot_signal_turnover(supplied)[0]
    assert compact.layout.meta["clock_source"] == "observed"
    assert full.layout.meta["clock_source"] == "supplied"
    assert full.layout.xaxis.title.text == "Supplied clock date"
    assert len(compact.data[0].x) == 11 and len(full.data[0].x) == 12
    assert [pd.Timestamp(value).value for value in full.data[0].x] == [day.value for day in dates]
    for trace in full.data:
        assert trace.y[5] is None and trace.y[6] is None
        assert trace.customdata[5][-1] == "empty_current"
        assert trace.customdata[6][-1] == "empty_previous"


def test_civil_daily_clock_retains_weekends_even_when_rotating_summary_means_agree():
    panel, observed, dates = captured_turnover()
    grid = pd.DataFrame({"date": pd.date_range(dates[0], dates[-1], freq="D")})
    supplied = TurnoverAnalysisResult.from_observations(
        panel.quantile_turnover(lags=(1, 2, 5), date_grid=grid),
        panel.rank_autocorrelation(lags=(1, 2, 5), date_grid=grid),
    )
    assert observed.get_dataframe("dated_autocorrelation").height == 36
    assert supplied.get_dataframe("dated_autocorrelation").height == 48
    assert observed.get_dataframe("autocorrelation")["n_dates"].to_list() == [11, 10, 7]
    assert supplied.get_dataframe("autocorrelation")["n_dates"].to_list() == [9, 6, 6]
    np.testing.assert_allclose(
        supplied.get_dataframe("autocorrelation")["autocorrelation"], [0, -1, 0], atol=1e-12
    )


@pytest.mark.parametrize("supplied", [False, True])
def test_unavailable_warmup_report_retains_requested_grid_without_inventing_values(supplied):
    from ml4t.diagnostic.visualization.signal import (
        generate_signal_turnover_report,
        plot_signal_turnover,
    )

    panel, _, dates = captured_turnover()
    data = panel.data
    warmup = data.select("date", "asset", "factor").with_columns(
        pl.col("factor").shift(100).over("asset")
    )
    empty = prepare_signal_panel(
        warmup, data.select("date", "asset", "1D"), quantiles=2, max_loss=1
    )
    grid = pd.DataFrame({"date": dates}) if supplied else None
    component = TurnoverAnalysisResult.from_observations(
        empty.quantile_turnover(lags=(1, 2, 5), quantiles=[1, 2], date_grid=grid),
        empty.rank_autocorrelation(lags=(1, 2, 5), date_grid=grid),
    )
    figures = plot_signal_turnover(component)
    assert len(figures) == (6 if supplied else 2)
    for figure in figures:
        assert any("unavailable" in note.text for note in figure.layout.annotations)
        for trace in figure.data:
            assert all(value is None for value in trace.y)
    assert "Signal turnover and rank persistence" in generate_signal_turnover_report(component)


@pytest.mark.parametrize("profile", ["rotating", "departures", "constant", "singleton"])
@pytest.mark.parametrize("nanoseconds", [False, True])
def test_owned_json_roundtrip_keeps_all_lag_dates_counts_and_undefined_series(profile, nanoseconds):
    from ml4t.diagnostic.visualization.signal import plot_signal_turnover

    _, component, dates = captured_turnover(profile, nanoseconds=nanoseconds)
    restored = TurnoverAnalysisResult.model_validate_json(component.model_dump_json())
    figures = plot_signal_turnover(restored)
    assert len(figures) == 6
    for figure in figures:
        kind, _ = _key(figure)
        for trace in figure.data:
            assert [pd.Timestamp(value).value for value in trace.x] == [
                date.value for date in dates
            ]
            assert trace.connectgaps is False
            if kind == "rank" and profile in {"constant", "singleton"}:
                assert all(value is None for value in trace.y)
        assert all("95% CI" not in note.text for note in figure.layout.annotations)
    for name in ["dated_turnover", "dated_autocorrelation", "turnover", "autocorrelation"]:
        assert restored.get_dataframe(name).equals(component.get_dataframe(name))


def test_complete_report_contains_all_quantile_and_rank_summaries_and_publishes_atomically(
    tmp_path,
):
    from ml4t.diagnostic.visualization.signal import generate_signal_turnover_report

    _, component, _ = captured_turnover()
    before = component.model_dump_json()
    target = tmp_path / "report.html"
    target.write_text("prior completed report")
    with pytest.raises(ValueError):
        generate_signal_turnover_report(component, output_file=target, theme="not-a-theme")
    assert target.read_text() == "prior completed report"
    directory = tmp_path / "existing-directory"
    directory.mkdir()
    with pytest.raises(OSError):
        generate_signal_turnover_report(component, output_file=directory)
    assert directory.is_dir() and not list(tmp_path.glob(".ml4t-events-*"))
    output = generate_signal_turnover_report(component, output_file=target)
    assert output == str(target.absolute())
    html = target.read_text()
    assert "All-quantile mean turnover" in html and "Mean rank autocorrelation" in html
    assert "entrants / current membership" in html
    assert "No confidence interval or half-life is inferred" in html
    assert component.model_dump_json() == before


@pytest.mark.parametrize("profile", ["rotating", "departures", "constant", "singleton"])
def test_complete_report_traces_and_summary_tables_match_actual_alphalens_on_shared_rows(profile):
    reference = pytest.importorskip("alphalens.performance")
    from importlib.metadata import version

    from ml4t.diagnostic.visualization.signal import plot_signal_turnover

    panel, _, dates = captured_turnover(profile)
    data = panel.data.to_pandas().rename(columns={"quantile": "factor_quantile"})
    data = data.set_index(["date", "asset"]).sort_index()
    assert data.index.levels[0].freq is None
    frozen = version("alphalens-reloaded") == "0.4.6"
    expected_dates = pd.date_range(dates[0], dates[-1], freq="D") if frozen else dates
    grid = pd.DataFrame({"date": expected_dates}) if frozen else None
    component = TurnoverAnalysisResult.from_observations(
        panel.quantile_turnover(lags=(1, 2, 5), date_grid=grid),
        panel.rank_autocorrelation(lags=(1, 2, 5), date_grid=grid),
    )
    figures = {_key(figure): figure for figure in plot_signal_turnover(component)}
    turnover_means = component.get_dataframe("turnover")
    rank_means = component.get_dataframe("autocorrelation")
    for lag in [1, 2, 5]:
        expected_rank = reference.factor_rank_autocorrelation(data, period=lag)
        rank = figures[("rank", lag)].data[0]
        assert [pd.Timestamp(value).value for value in rank.x] == [
            value.value for value in expected_dates
        ]
        assert [pd.Timestamp(value).value for value in rank.x] == [
            pd.Timestamp(value).value for value in expected_rank.index
        ]
        np.testing.assert_allclose(
            np.asarray(rank.y, dtype=float), expected_rank, atol=1e-12, equal_nan=True
        )
        mean = rank_means.filter(rank_means["lag"] == lag).row(0, named=True)
        np.testing.assert_allclose(
            float("nan") if mean["autocorrelation"] is None else mean["autocorrelation"],
            expected_rank.mean(),
            atol=1e-12,
            equal_nan=True,
        )
        assert mean["n_dates"] == expected_rank.notna().sum()
        for bucket in sorted(data["factor_quantile"].unique()):
            expected = reference.quantile_turnover(data["factor_quantile"], int(bucket), period=lag)
            actual = (
                component.get_dataframe("dated_turnover")
                .filter(
                    (component.get_dataframe("dated_turnover")["lag"] == lag)
                    & (component.get_dataframe("dated_turnover")["quantile"] == bucket)
                )
                .sort("date")
            )
            assert [pd.Timestamp(value).value for value in actual["date"].to_pandas()] == [
                pd.Timestamp(value).value for value in expected.index
            ]
            np.testing.assert_allclose(
                actual["turnover"].to_numpy(), expected, atol=1e-12, equal_nan=True
            )
            row = turnover_means.filter(
                (turnover_means["lag"] == lag) & (turnover_means["quantile"] == bucket)
            ).row(0, named=True)
            np.testing.assert_allclose(row["turnover"], expected.mean(), atol=1e-12)
            assert row["n_dates"] == expected.notna().sum()
    import alphalens.tears as reference_tears

    reference_tears.create_turnover_tear_sheet(data, turnover_periods=["1D", "2D", "5D"])
