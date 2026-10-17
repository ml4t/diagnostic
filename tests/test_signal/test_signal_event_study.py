"""Event studies must retain observed samples, elapsed bins and explicit rate units."""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml4t.diagnostic.signal import (
    SignalEventStudyResult,
    analyze_signal_event_study,
    prepare_signal_panel,
)


def _panel():
    dates = pd.to_datetime(["2025-01-02", "2025-01-06", "2025-01-08"], utc=True)
    index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "asset"])
    factor = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=index, name="factor")
    returns = pd.DataFrame({"3D": [0.03] * 6, "1D": [0.01] * 6}, index=index)
    return prepare_signal_panel(factor, returns, quantiles=1, max_loss=0)


def test_signal_event_study_uses_elapsed_bins_including_empty_bins_and_all_declared_events():
    result = analyze_signal_event_study(_panel(), rate_of_return=False, num_bars=3)
    assert result.event_distribution["n_events"].to_list() == [2, 0, 2, 2]
    assert result.event_distribution["date"].to_list() == list(
        pd.date_range("2025-01-02", "2025-01-08", freq="2D", tz="UTC")
    )
    assert result.metadata["distribution_unit"] == "elapsed"
    assert result.metadata["bin_width_ns"] == pd.Timedelta("2D").value
    stats = result.factor_statistics.row(0, named=True)
    assert stats["min"] == 1 and stats["max"] == 6 and stats["mean"] == 3.5
    assert stats["std"] == pytest.approx(np.sqrt(3.5))
    assert stats["count"] == 6 and stats["count_percent"] == 100
    assert result.event_returns is None
    assert result.metadata["event_paths_availability"] == "values_not_supplied"
    assert_frame_equal(result.prepared_events, _panel().data)
    assert (
        result.event_membership["bin_start"].to_list() == result.event_membership["date"].to_list()
    )


def test_event_rate_conversion_uses_first_supplied_horizon_without_relabelling_raw_statistics():
    result = analyze_signal_event_study(_panel(), rate_conversion_factors={"3D": 1, "1D": 3})
    assert result.metadata["base_period"] == "3D"
    assert result.metadata["conversion_ratios"] == {"3D": 1, "1D": 3}
    one = result.mean_returns.filter(pl.col("return_column") == "1D").row(0, named=True)
    assert one["mean"] == pytest.approx(0.01)
    assert one["display_mean"] == pytest.approx(1.01**3 - 1)
    assert one["std"] == pytest.approx(0)
    assert one["sample_unit"] == "date" and one["n_obs"] == 3
    assert result.return_observations.filter(pl.col("return_column") == "1D")[
        "n_obs"
    ].to_list() == [2, 2, 2]
    with pytest.raises(ValueError, match="rate_conversion_factors is required"):
        analyze_signal_event_study(_panel())


def test_event_study_serialization_owns_tables_and_keeps_undefined_rate_distinct_from_missing_mean():
    panel = _panel()
    frame = panel.data.to_pandas().set_index(["date", "asset"])
    factor = frame["factor"]
    returns = frame[["3D", "1D"]].copy()
    returns["1D"] = -1.5
    altered = prepare_signal_panel(factor, returns, quantiles=1, max_loss=0)
    result = analyze_signal_event_study(altered, rate_conversion_factors={"3D": 1, "1D": 0.5})
    undefined = result.mean_returns.filter(pl.col("return_column") == "1D")
    assert undefined["mean"][0] == -1.5 and undefined["display_mean"][0] is None
    assert undefined["display_status"][0] == "undefined_rate"
    restored = SignalEventStudyResult.from_json(result.to_json())
    assert_frame_equal(restored.mean_returns, result.mean_returns)
    copy = restored.factor_statistics
    copy[0, "mean"] = 999
    assert restored.factor_statistics["mean"][0] == 3.5


def test_single_date_event_distribution_is_one_observed_bin_not_a_zero_frequency_crash():
    frame = _panel().data.to_pandas()
    frame = frame[frame["date"] == frame["date"].iloc[0]].set_index(["date", "asset"])
    panel = prepare_signal_panel(frame["factor"], frame[["3D", "1D"]], quantiles=1, max_loss=0)
    result = analyze_signal_event_study(panel, rate_of_return=False)
    assert result.event_distribution["n_events"].to_list() == [2]
    assert result.metadata["bin_width_ns"] == 0


@pytest.mark.parametrize("num_bars", [True, 0, -1, 2.5])
def test_event_distribution_rejects_invalid_bin_requests(num_bars):
    with pytest.raises(ValueError, match="positive integer"):
        analyze_signal_event_study(_panel(), rate_of_return=False, num_bars=num_bars)


def test_factor_statistics_does_not_overflow_on_two_finite_large_events():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2025-01-02"], utc=True), ["A", "B"]], names=["date", "asset"]
    )
    factor = pd.Series([1e308, 1.1e308], index=index, name="factor")
    returns = pd.DataFrame({"1D": [0.0, 0.0]}, index=index)
    panel = prepare_signal_panel(factor, returns, quantiles=1, max_loss=0)
    stats = analyze_signal_event_study(panel, rate_of_return=False).factor_statistics.row(
        0, named=True
    )
    assert np.isfinite(stats["std"])
    assert stats["std"] == pytest.approx((1.1e308 - 1e308) / np.sqrt(2), rel=1e-12)
    assert stats["mean"] == pytest.approx(1.05e308, rel=1e-12)


def test_signal_event_study_composes_actual_returns_and_table_without_invented_paths(tmp_path):
    pytest.importorskip("plotly")
    from ml4t.diagnostic.visualization.signal import (
        generate_signal_event_study_report,
        plot_signal_event_study,
    )

    result = analyze_signal_event_study(
        _panel(), rate_conversion_factors={"3D": 1, "1D": 3}, num_bars=3
    )
    figures = plot_signal_event_study(result)
    assert len(figures) == 3
    assert list(figures[0].data[0].y) == [2, 0, 2, 2]
    assert list(figures[1].data[0].y) == pytest.approx([0.03])
    assert list(figures[1].data[1].y) == pytest.approx([1.01**3 - 1])
    assert list(figures[2].data[0].y) == pytest.approx([0.03] * 3)
    assert list(figures[2].data[1].y) == pytest.approx([1.01**3 - 1] * 3)
    target = tmp_path / "study.html"
    assert generate_signal_event_study_report(result, output_file=target) == str(target)
    html = target.read_text()
    assert html.count('class="plot-container"') == 3
    assert "Factor statistics" in html and "Event share (%)" in html
    assert "values_not_supplied" in html
    assert "Signal event study" in html


@pytest.mark.parametrize("rate", [False, True])
@pytest.mark.parametrize("num_bars", [3, 50])
@pytest.mark.parametrize(
    "with_values,with_window", [(False, False), (False, True), (True, False), (True, True)]
)
@pytest.mark.parametrize("quantiles", [1, 2])
def test_complete_event_study_matches_actual_reference_panels_and_intermediate_samples(
    rate, num_bars, with_values, with_window, quantiles, monkeypatch
):
    tears = pytest.importorskip("alphalens.tears")
    pytest.importorskip("plotly")
    import matplotlib.pyplot as plt

    from ml4t.diagnostic.visualization.signal import (
        generate_signal_event_study_report,
        plot_signal_event_study,
    )

    dates = pd.bdate_range("2025-01-02", periods=8, tz="UTC")
    prices = pd.DataFrame(
        np.arange(100.0, 106.0)
        * np.exp(np.arange(8)[:, None] * np.array([0.01, 0.02, -0.01, 0.03, -0.02, 0.005])),
        index=dates,
        columns=list("ABCDEF"),
    )
    factor = prices.iloc[[0, 2, 4]].stack().rename("factor")
    factor.index.names = ["date", "asset"]
    forward = pd.concat(
        {f"{period}D": (prices.shift(-period) / prices - 1).stack() for period in [3, 1]}, axis=1
    )
    forward.index.names = ["date", "asset"]
    panel = prepare_signal_panel(factor, forward, quantiles=quantiles, max_loss=0)
    frame = (
        panel.data.to_pandas()
        .rename(columns={"quantile": "factor_quantile"})
        .set_index(["date", "asset"])
    )
    captured = {}
    original_table = tears.plotting.plot_quantile_statistics_table
    original_distribution = tears.plotting.plot_events_distribution
    original_bar = tears.plotting.plot_quantile_returns_bar
    original_violin = tears.plotting.plot_quantile_returns_violin

    def stats(data, *args, **kwargs):
        captured["factors"] = original_table(data, return_df=True)
        return original_table(data, *args, **kwargs)

    def distribution(*args, **kwargs):
        axis = original_distribution(*args, **kwargs)
        captured["counts"] = [patch.get_height() for patch in axis.patches]
        captured["bin_labels"] = [label.get_text() for label in axis.get_xticklabels()]
        return axis

    def bars(data, *args, **kwargs):
        captured["means"] = data.copy(deep=True)
        axis = original_bar(data, *args, **kwargs)
        captured["bar_values"] = [patch.get_height() / 10000 for patch in axis.patches]
        return axis

    def violins(data, *args, **kwargs):
        captured["observations"] = data.copy(deep=True)
        return original_violin(data, *args, **kwargs)

    monkeypatch.setattr(tears.plotting, "plot_quantile_statistics_table", stats)
    monkeypatch.setattr(tears.plotting, "plot_events_distribution", distribution)
    monkeypatch.setattr(tears.plotting, "plot_quantile_returns_bar", bars)
    monkeypatch.setattr(tears.plotting, "plot_quantile_returns_violin", violins)
    values = prices if with_values else None
    window = (1, 2) if with_window else None
    tears.create_event_study_tear_sheet(
        frame, values, avgretplot=window, rate_of_ret=rate, n_bars=num_bars
    )
    result = analyze_signal_event_study(
        panel,
        values,
        input_kind="levels",
        event_window=window,
        rate_of_return=rate,
        rate_conversion_factors={"3D": 1, "1D": 3},
        num_bars=num_bars,
    )
    for row in result.factor_statistics.iter_rows(named=True):
        expected = captured["factors"].loc[row["quantile"]]
        for column in ["min", "max", "mean", "std", "count"]:
            assert row[column] == pytest.approx(expected[column], abs=1e-12)
        assert row["count_percent"] == pytest.approx(expected["count %"])
    assert result.event_distribution["n_events"].to_list() == captured["counts"]
    assert captured["bin_labels"] == [
        str(date) for date in result.event_distribution["date"].to_list()
    ]
    for row in result.mean_returns.iter_rows(named=True):
        np.testing.assert_allclose(
            row["display_mean"],
            captured["means"].loc[row["quantile"], row["return_column"]],
            atol=1e-12,
            rtol=1e-12,
        )
    expected_observations = (
        captured["observations"].reset_index().set_index(["date", "factor_quantile"])
    )
    for row in result.return_observations.iter_rows(named=True):
        np.testing.assert_allclose(
            row["display_mean"],
            expected_observations.loc[(row["date"], row["quantile"]), row["return_column"]],
            atol=1e-12,
            rtol=1e-12,
        )
    figures = plot_signal_event_study(result)
    expected_count = 3 + (1 + quantiles if with_values and with_window else 0)
    assert len(figures) == expected_count
    bars_figure = figures[-2]
    native_values = [value for trace in bars_figure.data for value in trace.y]
    np.testing.assert_allclose(native_values, captured["bar_values"], rtol=1e-12, atol=1e-12)
    for trace in figures[-1].data:
        sample = result.return_observations.filter(pl.col("return_column") == trace.name).sort(
            "quantile", "date"
        )
        np.testing.assert_allclose(trace.y, sample["display_mean"])
        assert len(trace.y) == sample.height
    html = generate_signal_event_study_report(result)
    assert html.count('class="plot-container"') == expected_count
    assert "Factor statistics" in html
    assert ("Aligned path availability" in html) == (not (with_values and with_window))
    plt.close("all")


def test_identity_rate_preserves_independent_tiny_return_instead_of_reference_rounding_to_zero():
    reference = pytest.importorskip("alphalens.utils")
    frame = _panel().data.to_pandas().set_index(["date", "asset"])
    returns = pd.DataFrame({"1D": [1e-20] * len(frame)}, index=frame.index)
    panel = prepare_signal_panel(frame["factor"], returns, quantiles=1, max_loss=0)
    result = analyze_signal_event_study(panel)
    assert result.mean_returns["display_mean"][0] == pytest.approx(1e-20, rel=1e-12, abs=0)
    assert reference.rate_of_return(pd.Series([1e-20], name="1D"), base_period="1D")[0] == 0


def test_reference_zero_span_histogram_failure_is_reconciled_to_independent_single_bin_count():
    reference = pytest.importorskip("alphalens.plotting")
    frame = _panel().data.to_pandas()
    frame = frame[frame["date"] == frame["date"].iloc[0]].set_index(["date", "asset"])
    with pytest.raises((ValueError, ZeroDivisionError)):
        reference.plot_events_distribution(frame["factor"])
    panel = prepare_signal_panel(frame["factor"], frame[["3D", "1D"]], quantiles=1, max_loss=0)
    result = analyze_signal_event_study(panel, rate_of_return=False)
    assert result.event_distribution["n_events"].to_list() == [2]


def test_reference_large_factor_dispersion_overflow_is_reconciled_to_two_point_expected_value():
    reference = pytest.importorskip("alphalens.plotting")
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2025-01-02"], utc=True), ["A", "B"]], names=["date", "asset"]
    )
    factor = pd.Series([1e308, 1.1e308], index=index, name="factor")
    returns = pd.DataFrame({"1D": [0.0, 0.0]}, index=index)
    panel = prepare_signal_panel(factor, returns, quantiles=1, max_loss=0)
    legacy_frame = (
        panel.data.to_pandas()
        .rename(columns={"quantile": "factor_quantile"})
        .set_index(["date", "asset"])
    )
    expected = (1.1e308 - 1e308) / np.sqrt(2)
    legacy = reference.plot_quantile_statistics_table(legacy_frame, return_df=True)
    assert not np.isfinite(legacy.loc[1, "std"])
    actual = analyze_signal_event_study(panel, rate_of_return=False).factor_statistics["std"][0]
    assert actual == pytest.approx(expected, rel=1e-12)
