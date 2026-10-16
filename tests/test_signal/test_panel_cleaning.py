"""Common supplied-return samples must be selected before factor bucketing."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.metrics import compute_forward_returns


def _inputs():
    dates = pd.bdate_range("2025-01-02", periods=6, tz="UTC")
    assets = ["A", "B", "C", "D"]
    wide = pd.DataFrame(
        {
            asset: [100.0 + 11 * i + 3 * day + ((day * (i + 2)) % 7) for day in range(6)]
            for i, asset in enumerate(assets)
        },
        index=dates,
    )
    index = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
    factor = (
        wide.pct_change(fill_method=None).stack(future_stack=True).reindex(index).rename("factor")
    )
    prices = (
        wide.stack(future_stack=True)
        .rename("close")
        .rename_axis(index=["date", "asset"])
        .reset_index()
    )
    forward = compute_forward_returns(prices, periods=[1, 2], group_col="asset")
    forward = forward.set_index(["date", "asset"])[["fwd_ret_1", "fwd_ret_2"]].rename(
        columns={"fwd_ret_1": "1D", "fwd_ret_2": "2D"}
    )
    return factor, forward, dates, assets


@pytest.mark.parametrize("representation", ["multiindex", "pandas", "polars"])
def test_causal_warmup_and_forward_tails_select_one_audited_sample_before_bucketing(representation):
    from ml4t.diagnostic.signal import prepare_signal_panel

    factor, forward, dates, assets = _inputs()
    original_factor, original_forward = factor.copy(deep=True), forward.copy(deep=True)
    if representation != "multiindex":
        factor, forward = factor.reset_index(), forward.reset_index()
        if representation == "polars":
            factor, forward = pl.from_pandas(factor), pl.from_pandas(forward)
    passed_factor = factor.clone() if representation == "polars" else factor.copy(deep=True)
    passed_forward = forward.clone() if representation == "polars" else forward.copy(deep=True)
    panel = prepare_signal_panel(factor, forward, quantiles=2, max_loss=0.5)
    data = panel.data.to_pandas().set_index(["date", "asset"]).sort_index()
    expected = [(day.value, asset) for day in dates[1:4] for asset in assets]
    assert [(pd.Timestamp(day).value, asset) for day, asset in data.index] == expected
    assert panel.metadata["input_rows"] == 24
    assert panel.metadata["retained_rows"] == 12
    assert panel.metadata["loss_fraction"] == 0.5
    assert panel.metadata["loss_by_stage"] == {
        "factor": 4,
        "returns": 8,
        "groups": 0,
        "bucketing": 0,
    }
    for day in dates[1:4]:
        values = data.loc[day, "factor"].to_numpy()
        expected_buckets = np.empty(4, dtype=int)
        expected_buckets[np.argsort(values)] = [1, 1, 2, 2]
        np.testing.assert_array_equal(data.loc[day, "quantile"], expected_buckets)
    np.testing.assert_allclose(data[["1D", "2D"]], original_forward.reindex(data.index))
    if representation == "multiindex":
        pd.testing.assert_series_equal(factor, original_factor)
        pd.testing.assert_frame_equal(forward, original_forward)
    elif representation == "pandas":
        pd.testing.assert_frame_equal(factor, passed_factor)
        pd.testing.assert_frame_equal(forward, passed_forward)
    else:
        assert factor.equals(passed_factor)
        assert forward.equals(passed_forward)


def test_loss_limit_uses_original_factor_denominator_and_includes_exact_boundary():
    from ml4t.diagnostic.signal import prepare_signal_panel

    factor, forward, _, _ = _inputs()
    assert prepare_signal_panel(factor, forward, quantiles=2, max_loss=0.5).data.height == 12
    with pytest.raises(ValueError, match="max_loss"):
        prepare_signal_panel(factor, forward, quantiles=2, max_loss=np.nextafter(0.5, 0.0))


def test_group_relative_buckets_use_dated_groups_on_the_same_complete_sample():
    from ml4t.diagnostic.signal import prepare_signal_panel

    factor, forward, dates, assets = _inputs()
    groups = pd.Series(
        ["g1" if asset in ["A", "B"] else "g2" for _ in dates for asset in assets],
        index=factor.index,
    )
    panel = prepare_signal_panel(
        factor,
        forward,
        quantiles=2,
        groups=groups,
        group_labels={"g1": "sector one", "g2": "sector two"},
        binning_by_group=True,
        max_loss=0.5,
    )
    data = panel.data
    assert data.height == 12
    assert set(data["group"].to_list()) == {"sector one", "sector two"}
    for group in data.partition_by(["date", "group"]):
        assert sorted(group["quantile"].to_list()) == [1, 2]


def test_constant_cross_section_is_a_bucket_loss_not_an_unreported_sample_change():
    from ml4t.diagnostic.signal import prepare_signal_panel

    factor, forward, dates, _ = _inputs()
    factor.loc[dates[2]] = factor.loc[dates[2]].mean()
    panel = prepare_signal_panel(factor, forward, quantiles=2, max_loss=0.7)
    assert panel.data.height == 8
    assert panel.metadata["loss_by_stage"] == {
        "factor": 4,
        "returns": 8,
        "groups": 0,
        "bucketing": 4,
    }
    assert panel.metadata["bucket_failures"][0]["rows"] == 4
    with pytest.raises((ValueError, pl.exceptions.DuplicateError), match="edges"):
        prepare_signal_panel(factor, forward, quantiles=2, max_loss=0.0)


def test_panel_and_analysis_own_cleaned_values_and_audit_through_json(tmp_path):
    from ml4t.diagnostic import SignalResult
    from ml4t.diagnostic.signal import prepare_signal_panel

    factor, forward, _, _ = _inputs()
    panel = prepare_signal_panel(factor, forward, quantiles=2, max_loss=0.5)
    before = panel.data
    accessed = panel.data
    accessed.replace_column(accessed.get_column_index("factor"), accessed["factor"] * 0.0)
    assert panel.data.equals(before)
    factor.iloc[:] = -10.0
    forward.iloc[:, :] = -1.0
    audit = panel.metadata
    audit["loss_by_stage"]["factor"] = 999
    assert panel.data.equals(before)
    assert panel.metadata["loss_by_stage"]["factor"] == 4
    result = panel.analyze(min_assets=2)
    assert result.n_dates == 3 and result.n_assets == 4
    assert result.preparation_metadata["cleaning"]["input_rows"] == 24
    assert sum(result.count_by_quantile.values()) == 12
    path = tmp_path / "panel.json"
    result.to_json(str(path))
    restored = SignalResult.from_json(str(path))
    assert restored.prepared_data.equals(result.prepared_data)
    assert restored.preparation_metadata["cleaning"] == panel.metadata
