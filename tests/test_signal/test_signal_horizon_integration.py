"""Prepared outcomes retain their selectors and actual buckets through analysis."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic import analyze_prepared_signal
from ml4t.diagnostic.signal import SignalResult, prepare_signal_panel


def _supplied_panel(horizons=("30m",), *, sparse=False):
    dates = pd.date_range("2025-07-07T00:30:00Z", periods=4, freq="1ns")
    index = pd.MultiIndex.from_product([dates, ["A", "B", "C", "D"]], names=["date", "asset"])
    factor = pd.Series(np.tile([-3.0, -1.0, 1.0, 3.0], 4), index=index, name="factor")
    outcomes = pd.DataFrame(
        {
            name: np.tile([-0.02, 0.0, 0.02, 0.04], 4) * (column + 1)
            + np.repeat([0.0, 0.01, -0.01, 0.02], 4)
            for column, name in enumerate(horizons)
        },
        index=index,
    )
    original_factor = factor.copy(deep=True)
    original_outcomes = outcomes.copy(deep=True)
    panel = prepare_signal_panel(
        factor,
        outcomes,
        quantiles=None if sparse else 2,
        bins=[-10.0, -5.0, 0.0, 5.0] if sparse else None,
        return_columns=horizons,
        max_loss=0,
    )
    pd.testing.assert_series_equal(factor, original_factor)
    pd.testing.assert_frame_equal(outcomes, original_outcomes)
    return panel, factor, outcomes


@pytest.mark.parametrize("horizon", ["30m", "1ns", "1D1h", "realized_excess", "21"])
@pytest.mark.parametrize("sparse", [False, True])
def test_cleaner_produced_subday_and_custom_buckets_reach_analysis_without_relabeling(
    horizon, sparse, tmp_path
):
    panel, factor, outcomes = _supplied_panel((horizon,), sparse=sparse)
    before = panel.data
    expected_ids = [2, 3] if sparse else [1, 2]
    assert sorted(before["quantile"].unique()) == expected_ids
    result = panel.analyze(min_assets=4)
    assert result.horizon_labels == (horizon,)
    assert list(result.ic) == [horizon] and result.ic[horizon] == pytest.approx(1.0)
    assert result.quantile_ids == tuple(expected_ids)
    assert result.quantiles == 2
    assert list(result.quantile_returns[horizon]) == expected_ids
    assert result.quantile_returns[horizon] == pytest.approx(
        {expected_ids[0]: -0.005, expected_ids[1]: 0.035}
    )
    assert result.spread[horizon] == pytest.approx(0.04)
    assert result.prepared_data.equals(before)
    assert panel.data.equals(before)
    np.testing.assert_array_equal(factor, np.tile([-3.0, -1.0, 1.0, 3.0], 4))
    np.testing.assert_array_equal(result.prepared_data[horizon], outcomes[horizon])
    assert result.to_ic_result(period=horizon).ic_mean == {horizon: pytest.approx(1.0)}
    assert result.to_ic_result().get_ic(horizon) == pytest.approx(1.0)
    quantile = result.to_quantile_result()
    assert quantile.periods == [horizon]
    assert quantile.quantile_labels == [f"Q{bucket}" for bucket in expected_ids]
    assert quantile.get_quantile_returns(horizon) == pytest.approx(
        {f"Q{expected_ids[0]}": -0.005, f"Q{expected_ids[1]}": 0.035}
    )
    assert quantile.get_spread(horizon)[0] == pytest.approx(0.04)
    assert set(quantile.count_by_quantile) == set(quantile.quantile_labels)
    assert result.to_tear_sheet().quantile_analysis.quantile_labels == quantile.quantile_labels
    assert horizon in result.summary()
    target = tmp_path / "result.json"
    result.to_json(str(target))
    restored = SignalResult.from_json(str(target))
    assert restored.horizon_labels == (horizon,)
    assert restored.quantile_ids == tuple(expected_ids)
    assert restored.prepared_data.equals(before)
    assert restored.quantile_observations.equals(result.quantile_observations)


def test_sparse_bucket_empirical_plots_use_real_dated_means_not_contiguous_replacements(tmp_path):
    from ml4t.diagnostic.visualization.signal import (
        plot_quantile_returns_bar,
        plot_quantile_returns_violin,
    )

    panel, _, _ = _supplied_panel(sparse=True)
    result = panel.analyze(min_assets=4)
    quantile = result.to_quantile_result()
    bar = plot_quantile_returns_bar(quantile, period="30m")
    assert list(bar.data[0].x) == ["Q2", "Q3"]
    np.testing.assert_allclose(bar.data[0].y, [-0.005, 0.035])
    violin = plot_quantile_returns_violin(quantile, period="30m")
    traces = [trace for trace in violin.data if trace.type == "violin"]
    assert len(traces) == 2
    np.testing.assert_allclose(sorted(traces[0].y), [-0.02, -0.01, 0.0, 0.01])
    np.testing.assert_allclose(sorted(traces[1].y), [0.02, 0.03, 0.04, 0.05])
    raw = plot_quantile_returns_violin(quantile, period="30m", factor_data=result.prepared_data)
    assert len(raw.data[0].y) == len(raw.data[1].y) == 8
    from ml4t.diagnostic.signal._report import generate_html

    target = tmp_path / "report.html"
    generate_html(result, str(target))
    assert target.stat().st_size > 1000


def test_explicit_return_selectors_preserve_requested_order_and_owned_nanosecond_keys():
    panel, _, _ = _supplied_panel(("1D1h", "1ns", "30m"), sparse=True)
    result = analyze_prepared_signal(
        panel.data, return_columns=("30m", "1D1h", "1ns"), min_assets=4
    )
    assert result.horizon_labels == ("30m", "1D1h", "1ns")
    assert list(result.ic) == ["30m", "1D1h", "1ns"]
    np.testing.assert_array_equal(
        result.prepared_data["date"].cast(pl.Int64), panel.data["date"].cast(pl.Int64)
    )
    assert result.preparation_metadata["horizon_order"] == ["30m", "1D1h", "1ns"]


def test_subday_horizons_do_not_become_daily_turnover_lags():
    panel, _, _ = _supplied_panel(("30m", "1ns", "1D1h"), sparse=True)
    result = panel.analyze(min_assets=4, compute_turnover_flag=True)
    assert result.turnover == {}
    assert result.dated_turnover is None
    assert result.preparation_metadata["automatic_turnover_lags"] == []
    mixed, _, _ = _supplied_panel(("30m", "1D", "2D"), sparse=True)
    result = mixed.analyze(min_assets=4, compute_turnover_flag=True)
    assert set(result.turnover) == {"1D", "2D"}
    assert set(result.dated_turnover["quantile"]) == {2, 3}
    assert set(result.dated_turnover["lag"]) == {1, 2}


@pytest.mark.parametrize("columns", [[], ["30m", "30m"], ["missing"], ["factor"], "30m"])
def test_invalid_supplied_selectors_reject_without_modifying_the_prepared_sample(columns):
    panel, _, _ = _supplied_panel()
    values = panel.data
    before = values.clone()
    with pytest.raises((ValueError, TypeError)):
        analyze_prepared_signal(values, return_columns=columns, min_assets=4)
    assert values.equals(before)


def test_ambiguous_period_and_explicit_selector_arguments_are_not_silently_prioritized():
    panel, _, _ = _supplied_panel()
    with pytest.raises(ValueError, match="periods.*return_columns"):
        analyze_prepared_signal(panel.data, periods=(1,), return_columns=("30m",))


@pytest.mark.parametrize("horizon", ["30m", "1ns", "1D1h"])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_corrected_alphalens_shared_horizon_membership_and_distinct_reduction_definitions(
    horizon, sparse, missing
):
    utils = pytest.importorskip("alphalens.utils")
    performance = pytest.importorskip("alphalens.performance")
    from importlib.metadata import version

    if version("alphalens-reloaded") != "0.5.0":
        pytest.skip("this regression pins the corrected Alphalens0.5.0 profile")
    _, factor, outcomes = _supplied_panel((horizon,), sparse=sparse)
    if missing:
        outcomes.iloc[0, 0] = np.nan
    bins = [-10, -5, 0, 5] if sparse else [-4, 0, 4]
    native = prepare_signal_panel(factor, outcomes, quantiles=None, bins=bins, max_loss=1 / 16)
    reference = utils.get_clean_factor(factor, outcomes, quantiles=None, bins=bins, max_loss=1 / 16)
    values = native.data.to_pandas().set_index(["date", "asset"]).sort_index()
    compared = reference.rename(columns={"factor_quantile": "quantile"}).sort_index()
    pd.testing.assert_index_equal(values.index, compared.index, check_names=True)
    pd.testing.assert_frame_equal(
        values[["factor", horizon, "quantile"]],
        compared[["factor", horizon, "quantile"]],
        check_dtype=False,
        check_freq=False,
    )
    assert values.shape[0] == (15 if missing else 16)
    assert native.metadata["loss_by_stage"]["returns"] == int(missing)
    result = native.analyze(min_assets=2)
    reference_ic = performance.factor_information_coefficient(reference)
    np.testing.assert_allclose(result.ic_series[horizon], reference_ic[horizon], atol=1e-12)
    native_dates = pd.DatetimeIndex(pd.to_datetime(result.ic_dates[horizon], format="mixed"))
    np.testing.assert_array_equal(
        native_dates.as_unit("ns").asi8, reference_ic.index.as_unit("ns").asi8
    )
    means, _ = performance.mean_return_by_quantile(reference, demeaned=False)
    low, high = result.quantile_ids
    assert result.quantile_returns[horizon][high] == pytest.approx(0.035)
    assert means.loc[high, horizon] == pytest.approx(0.035)
    # Removing the actual first low-asset outcome makes date counts unequal.
    # Native pools seven asset outcomes; Alphalens gives all four dates equal weight.
    assert result.quantile_returns[horizon][low] == pytest.approx(-0.02 / 7 if missing else -0.005)
    assert means.loc[low, horizon] == pytest.approx(-0.0025 if missing else -0.005)
