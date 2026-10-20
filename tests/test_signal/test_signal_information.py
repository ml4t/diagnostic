"""Complete information reports retain actual global/group samples and undefined states."""

import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from scipy import stats

from ml4t.diagnostic.signal import (
    SignalInformationResult,
    analyze_signal_information,
    prepare_signal_panel,
)


def _inputs():
    recipe = Path(__file__).resolve().parents[2] / "examples/legacy_migration_comparison.py"
    return runpy.run_path(str(recipe))["_signal_information_inputs"]()


@pytest.mark.parametrize("adjusted", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
def test_information_facade_retains_both_global_and_requested_unequal_group_memberships(
    adjusted, grouped
):
    factor, forward, groups = _inputs()
    panel = prepare_signal_panel(factor, forward, groups=groups, quantiles=3, max_loss=0)
    result = analyze_signal_information(panel, group_adjust=adjusted, by_group=grouped)
    global_ic, group_ic = result.get_table("ic_by_date"), result.get_table("group_ic_by_date")
    assert global_ic.height == 64 and global_ic["n_obs"].to_list() == [7] * 64
    assert global_ic["status"].to_list() == ["defined"] * 64
    assert result.get_table("prepared_data").height == 224
    assert group_ic.height == (128 if grouped else 0)
    if grouped:
        for row in group_ic.iter_rows(named=True):
            small = 3 if row["date"].day % 2 else 2
            assert row["n_obs"] == (small if row["group"] == "small" else 7 - small)
        assert set(result.get_table("mean_group_ic")["group"]) == {"small", "large"}
    else:
        assert result.get_table("mean_group_ic").is_empty()
    monthly = result.get_table("monthly_ic")
    assert set(monthly["month"]) == {1, 2, 3}
    assert monthly["n_dates"].sum() == 64
    for row in result.get_table("summary").iter_rows(named=True):
        values = global_ic.filter(pl.col("return_column") == row["return_column"])["ic"].to_numpy()
        expected = stats.ttest_1samp(values, 0)
        assert row["mean_ic"] == pytest.approx(np.mean(values), abs=1e-14)
        assert row["std_ic"] == pytest.approx(np.std(values, ddof=1), abs=1e-14)
        assert row["t_stat"] == pytest.approx(expected.statistic, abs=1e-12)
        assert row["p_value"] == pytest.approx(expected.pvalue, abs=1e-12)
        assert row["n_dates"] == row["n_defined"] == 32
    adapted = result.to_ic_result()
    assert len(adapted.dates) == 32 and all(
        len(values) == 32 for values in adapted.ic_by_date.values()
    )


def test_information_export_owns_original_input_keys_groups_and_every_intermediate_table():
    factor, forward, groups = _inputs()
    panel = prepare_signal_panel(factor, forward, groups=groups, quantiles=3, max_loss=0)
    result = analyze_signal_information(panel, group_adjust=True, by_group=True)
    restored = SignalInformationResult.from_json(result.to_json())
    for name in [
        "prepared_data",
        "ic_by_date",
        "group_ic_by_date",
        "summary",
        "monthly_ic",
        "mean_group_ic",
    ]:
        assert_frame_equal(result.get_table(name), restored.get_table(name))
    frame = result.get_table("prepared_data")
    frame[0, "factor"] = 999
    factor.iloc[0] = -999
    groups.iloc[0] = "changed"
    metadata = result.metadata
    metadata["return_columns"].append("invented")
    assert_frame_equal(panel.data, result.get_table("prepared_data"))
    assert result.metadata["return_columns"] == ["1D", "5D"]
    with pytest.raises(ValueError, match="Unknown information table"):
        result.get_table("not_a_table")


def test_information_chart_adapter_keeps_price_observation_keys_below_microsecond_resolution():
    factor, forward, groups = _inputs()
    original_dates = factor.index.get_level_values("date").unique()
    clock = pd.date_range("2025-01-20", periods=32, freq="100ns", tz="UTC")
    mapping = dict(zip(original_dates, clock, strict=True))
    for table in [factor, forward, groups]:
        table.index = pd.MultiIndex.from_tuples(
            [(mapping[date], asset) for date, asset in table.index], names=["date", "asset"]
        )
    forward = forward.rename(columns={"1D": "next_quote", "5D": "fifth_quote"})
    panel = prepare_signal_panel(factor, forward, groups=groups, quantiles=3, max_loss=0)
    result = analyze_signal_information(panel, by_group=True)
    assert result.get_table("ic_by_date")["date"].n_unique() == 32
    model = result.to_ic_result()
    assert list(pd.DatetimeIndex(model.dates).as_unit("ns").asi8) == list(clock.as_unit("ns").asi8)
    assert set(model.ic_by_date) == {"next_quote", "fifth_quote"}


def test_information_keeps_unavailable_dates_and_counts_instead_of_inventing_zero_ic():
    dates = pd.bdate_range("2025-01-02", periods=25, tz="UTC")
    prices = pd.DataFrame(np.arange(1, 13)[None, :] * 2.0 ** np.arange(25)[:, None], index=dates)
    factors = (
        pd.DataFrame(np.tile(np.arange(1, 13), (24, 1)), index=dates[:-1]).stack().rename("factor")
    )
    factors.index = factors.index.set_names(["date", "asset"])
    forward = (prices.shift(-1) / prices - 1).stack().to_frame("one_observation")
    forward.index = forward.index.set_names(["date", "asset"])
    panel = prepare_signal_panel(factors, forward, quantiles=3, max_loss=0)
    result = analyze_signal_information(panel)
    observations = result.get_table("ic_by_date")
    assert observations.height == 24 and observations["n_obs"].to_list() == [12] * 24
    assert observations["ic"].null_count() == 24
    assert observations["status"].to_list() == ["constant_outcome"] * 24
    summary = result.get_table("summary").row(0, named=True)
    assert summary["n_defined"] == 0 and summary["n_dates"] == 24
    assert summary["mean_ic"] is summary["t_stat"] is summary["p_value"] is None
    assert result.get_table("monthly_ic")["mean_ic"].null_count() > 0
    adapted = result.to_ic_result()
    assert len(adapted.dates) == 24 and np.isnan(adapted.ic_by_date["one_observation"]).all()


@pytest.mark.parametrize("adjusted", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
def test_complete_information_report_matches_actual_reference_sheets_and_their_source_samples(
    adjusted, grouped
):
    reference = pytest.importorskip("alphalens.performance")
    plotting = pytest.importorskip("alphalens.plotting")
    tears = pytest.importorskip("alphalens.tears")
    utils = pytest.importorskip("alphalens.utils")
    from ml4t.diagnostic.visualization.signal import (
        generate_signal_information_report,
        plot_signal_information,
    )

    factor, forward, groups = _inputs()
    frame = utils.get_clean_factor(factor, forward, groupby=groups, quantiles=3, max_loss=0)
    tears.create_information_tear_sheet(frame, group_neutral=adjusted, by_group=grouped)
    panel = prepare_signal_panel(factor, forward, groups=groups, quantiles=3, max_loss=0)
    result = analyze_signal_information(panel, group_adjust=adjusted, by_group=grouped)
    prepared = result.get_table("prepared_data").to_pandas().set_index(["date", "asset"])
    assert set(prepared.index) == set(frame.index) and len(prepared) == len(frame) == 224
    for column in ["factor", "1D", "5D", "group"]:
        pd.testing.assert_series_equal(
            prepared[column].sort_index(),
            frame[column].sort_index(),
            check_names=False,
            check_dtype=False,
            check_categorical=False,
            check_freq=False,
        )
    pd.testing.assert_series_equal(
        prepared["quantile"].sort_index(),
        frame["factor_quantile"].sort_index(),
        check_names=False,
        check_dtype=False,
        check_freq=False,
    )
    for by_group in [False, True] if grouped else [False]:
        expected = reference.factor_information_coefficient(
            frame, group_adjust=adjusted, by_group=by_group
        )
        if by_group:
            expected = expected.reset_index()
            expected["group"] = expected["group"].astype(object)
            expected = expected.set_index(["date", "group"])
        actual = result.get_table("group_ic_by_date" if by_group else "ic_by_date").to_pandas()
        keys = ["date", "group"] if by_group else ["date"]
        actual = actual.pivot(index=keys, columns="return_column", values="ic").sort_index()
        pd.testing.assert_frame_equal(
            actual,
            expected.sort_index(),
            check_names=False,
            check_dtype=False,
            check_freq=False,
            atol=1e-12,
            rtol=1e-12,
        )
    expected = reference.factor_information_coefficient(frame, group_adjust=adjusted)
    summary = plotting.plot_information_table(expected, return_df=True)
    native = result.get_table("summary").to_pandas().set_index("return_column")
    for native_name, reference_name in [
        ("mean_ic", "IC Mean"),
        ("std_ic", "IC Std."),
        ("risk_adjusted_ic", "Risk-Adjusted IC"),
        ("t_stat", "t-stat(IC)"),
        ("p_value", "p-value(IC)"),
        ("skew", "IC Skew"),
        ("kurtosis", "IC Kurtosis"),
    ]:
        np.testing.assert_allclose(
            native.loc[expected.columns, native_name],
            summary.loc[expected.columns, reference_name],
            atol=1e-12,
            rtol=1e-12,
        )
    figures = plot_signal_information(result)
    assert len(figures) == (7 if grouped else 8)
    for position, column in enumerate(expected.columns):
        time_series, histogram, qq = figures[3 * position : 3 * position + 3]
        np.testing.assert_allclose(time_series.data[0].y, expected[column], atol=1e-12)
        rolling = next(trace for trace in time_series.data if "Rolling" in trace.name)
        np.testing.assert_allclose(
            rolling.y, expected[column].rolling(22).mean(), atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(histogram.data[0].x, expected[column].dropna(), atol=1e-12)
        axes = plotting.plot_ic_qq(expected[[column]])
        np.testing.assert_allclose(qq.data[0].x, axes[0].lines[0].get_xdata(), atol=1e-12)
        np.testing.assert_allclose(qq.data[0].y, axes[0].lines[0].get_ydata(), atol=1e-12)
    if grouped:
        expected_groups = reference.mean_information_coefficient(
            frame, group_adjust=adjusted, by_group=True
        )
        for trace in figures[-1].data:
            np.testing.assert_allclose(
                trace.y, expected_groups.loc[list(trace.x), trace.name], atol=1e-12
            )
    else:
        expected_months = reference.mean_information_coefficient(
            frame, group_adjust=adjusted, by_time=pd.offsets.MonthEnd()
        )
        for row in result.get_table("monthly_ic").iter_rows(named=True):
            selected = expected_months[
                (expected_months.index.year == row["year"])
                & (expected_months.index.month == row["month"])
            ][row["return_column"]]
            assert len(selected) == 1 and row["mean_ic"] == pytest.approx(
                selected.iloc[0], abs=1e-12
            )
    html = generate_signal_information_report(result)
    assert html.count('class="plot-container"') == len(figures)
    assert "Information analysis" in html and "Defined ICs" in html


@pytest.mark.parametrize("distribution", [stats.norm, stats.t])
def test_fitted_qq_matches_actual_reference_distribution_without_changing_native_default(
    distribution,
):
    plotting = pytest.importorskip("alphalens.plotting")
    from ml4t.diagnostic.visualization.signal import plot_ic_qq

    factor, forward, groups = _inputs()
    result = analyze_signal_information(
        prepare_signal_panel(factor, forward, groups=groups, quantiles=3, max_loss=0)
    )
    model = result.to_ic_result()
    frame = pd.DataFrame(model.ic_by_date, index=pd.to_datetime(model.dates))[["1D"]]
    axes = plotting.plot_ic_qq(frame, theoretical_dist=distribution)
    figure = plot_ic_qq(model, period="1D", fitted=True, theoretical_dist=distribution)
    np.testing.assert_allclose(figure.data[0].x, axes[0].lines[0].get_xdata(), atol=1e-12)
    np.testing.assert_allclose(figure.data[0].y, axes[0].lines[0].get_ydata(), atol=1e-12)
    native = plot_ic_qq(model, period="1D")
    expected_native = stats.norm.ppf(
        (np.arange(1, 33) - 0.5) / 32, loc=np.mean(frame["1D"]), scale=np.std(frame["1D"], ddof=1)
    )
    np.testing.assert_allclose(native.data[0].x, expected_native, atol=1e-12)
    assert native.layout.meta["qq_method"] == "native_midpoint_sample_std_normal"


def test_information_report_preserves_existing_target_on_real_filesystem_failure(tmp_path):
    from ml4t.diagnostic.visualization.signal import generate_signal_information_report

    factor, forward, groups = _inputs()
    result = analyze_signal_information(
        prepare_signal_panel(factor, forward, groups=groups, quantiles=3, max_loss=0)
    )
    directory = tmp_path / "user-owned"
    directory.mkdir()
    marker = directory / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(OSError):
        generate_signal_information_report(result, output_file=directory)
    assert marker.read_text() == "keep" and list(tmp_path.iterdir()) == [directory]
