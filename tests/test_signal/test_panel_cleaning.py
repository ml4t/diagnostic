"""Common supplied-return samples must be selected before factor bucketing."""

import base64
import math
from io import BytesIO

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.metrics import compute_forward_returns


def _inputs(nanosecond=False, zero_mark=False):
    dates = pd.bdate_range("2025-01-02", periods=6, tz="UTC")
    if nanosecond:
        dates = pd.DatetimeIndex(
            [dates[0] + pd.Timedelta(100 * offset, unit="ns") for offset in range(6)]
        )
    assets = ["A", "B", "C", "D"]
    wide = pd.DataFrame(
        {
            asset: [100.0 + 11 * i + 3 * day + ((day * (i + 2)) % 7) for day in range(6)]
            for i, asset in enumerate(assets)
        },
        index=dates,
    )
    if zero_mark:
        wide.loc[dates[2], "C"] = 0.0
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
    if nanosecond:
        forward = forward.rename(columns={"1D": "1bar", "2D": "2bar"})
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


@pytest.mark.parametrize(
    "case",
    [
        "probabilities",
        "partial_probabilities",
        "width",
        "edges",
        "infinite_edges",
        "zero_q4",
        "zero_q3",
        "zero_width",
    ],
)
def test_supported_bucket_options_preserve_independent_membership_and_account_for_outside_edges(
    case,
):
    from ml4t.diagnostic.signal import prepare_signal_panel

    factor, forward, dates, _ = _inputs()
    factor = factor - factor.groupby(level="date").transform("mean")
    options = {"quantiles": 2}
    expected_sorted = [1, 1, 2, 2]
    expected_count = 2
    if case == "probabilities":
        options["quantiles"] = [0.0, 0.25, 0.75, 1.0]
        expected_sorted, expected_count = [1, 2, 2, 3], 3
    elif case == "partial_probabilities":
        options["quantiles"] = [0.25, 0.75]
        expected_sorted, expected_count = [1, 1], 1
    elif case in ["width", "edges", "infinite_edges", "zero_width"]:
        options = {
            "quantiles": None,
            "bins": 2
            if case == "width"
            else [-1.0, 0.0, 1.0]
            if case == "edges"
            else [-np.inf, 0.0, np.inf]
            if case == "infinite_edges"
            else 4,
        }
        if case == "zero_width":
            options["zero_aware"] = True
            expected_sorted, expected_count = [1, 2, 3, 4], 4
    else:
        options = {"quantiles": 4 if case == "zero_q4" else 3, "zero_aware": True}
        if case == "zero_q4":
            expected_sorted, expected_count = [1, 2, 3, 4], 4
    panel = prepare_signal_panel(factor, forward, max_loss=0.8, **options)
    data = panel.data.to_pandas().set_index(["date", "asset"])
    assert len(data) == (6 if case == "partial_probabilities" else 12)
    assert panel.metadata["bucket_count"] == expected_count
    assert panel.metadata["bucket_method"] == ("pandas.cut" if "bins" in options else "pandas.qcut")
    assert panel.metadata["loss_by_stage"] == {
        "factor": 4,
        "returns": 8,
        "groups": 0,
        "bucketing": 6 if case == "partial_probabilities" else 0,
    }
    for day in dates[1:4]:
        actual = data.loc[day].sort_values("factor")
        available = factor.loc[day].sort_values()
        if case == "partial_probabilities":
            available = available.iloc[1:3]
        assert actual.index.tolist() == available.index.tolist()
        if case == "width":
            midpoint = (available.iloc[0] + available.iloc[-1]) / 2
            expected = np.where(available <= midpoint, 1, 2)
        else:
            expected = expected_sorted
        np.testing.assert_array_equal(actual["quantile"], expected)


@pytest.mark.parametrize("labels", [None, {"g1": "sector one", "g2": "sector two"}])
def test_real_dated_group_gap_and_unbucketable_remainder_have_separate_losses(labels):
    from ml4t.diagnostic.signal import prepare_signal_panel

    factor, forward, dates, assets = _inputs()
    groups = pd.Series(
        ["g1" if asset in ["A", "B"] else "g2" for _ in dates for asset in assets],
        index=factor.index,
    )
    groups = groups.drop((dates[2], "A")).reindex(factor.index)
    panel = prepare_signal_panel(
        factor,
        forward,
        quantiles=2,
        groups=groups,
        group_labels=labels,
        binning_by_group=True,
        max_loss=0.6,
    )
    assert panel.data.height == 10
    assert panel.metadata["loss_by_stage"] == {
        "factor": 4,
        "returns": 8,
        "groups": 1,
        "bucketing": 1,
    }
    assert panel.metadata["bucket_failures"][0]["rows"] == 1


@pytest.mark.parametrize("empty_source", ["factor", "returns", "static_groups"])
def test_explicit_total_loss_keeps_an_owned_empty_panel_not_a_schema_error(empty_source):
    from ml4t.diagnostic.signal import prepare_signal_panel

    factor, forward, _, _ = _inputs()
    groups = None
    if empty_source in ["factor", "static_groups"]:
        factor = factor.groupby(level="asset").transform(lambda series: series.rolling(20).mean())
        if empty_source == "static_groups":
            groups = {}
    else:
        forward = forward.shift(-len(forward))
    panel = prepare_signal_panel(factor, forward, quantiles=2, groups=groups, max_loss=1.0)
    assert panel.data.height == 0
    assert sum(panel.metadata["loss_by_stage"].values()) == 24
    with pytest.raises(ValueError, match="max_loss"):
        prepare_signal_panel(
            factor, forward, quantiles=2, groups=groups, max_loss=np.nextafter(1.0, 0.0)
        )
    with pytest.raises(ValueError, match="nonempty"):
        panel.analyze(min_assets=2)


def test_submicrosecond_observed_bar_keys_are_preserved_without_claiming_day_horizons(tmp_path):
    from ml4t.diagnostic.signal import prepare_signal_panel

    factor, forward, dates, assets = _inputs(nanosecond=True)
    panel = prepare_signal_panel(factor, forward, quantiles=2, max_loss=0.5)
    data = panel.data
    assert data.height == 12
    assert data["date"].dtype == pl.Datetime("ns", "UTC")
    expected = [day.value for day in dates[1:4] for _ in assets]
    assert data["date"].cast(pl.Int64).to_list() == expected
    assert panel.metadata["return_columns"] == ["1bar", "2bar"]
    path = tmp_path / "existing.json"
    path.write_text("existing report")
    with pytest.raises(ValueError, match="horizon labels"):
        panel.analyze(min_assets=2).to_json(str(path))
    assert path.read_text() == "existing report"


@pytest.mark.parametrize("side", ["factor", "returns", "groups"])
def test_overlapping_input_batches_cannot_be_hidden_as_cleaning_losses(side):
    from ml4t.diagnostic.signal import prepare_signal_panel

    factor, forward, _, _ = _inputs()
    groups = pd.Series("sector", index=factor.index)
    if side == "factor":
        factor = pd.concat([factor, factor.iloc[[0]]])
    elif side == "returns":
        forward = pd.concat([forward, forward.iloc[[0]]])
    else:
        groups = pd.concat([groups, groups.iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate"):
        prepare_signal_panel(factor, forward, quantiles=2, groups=groups, max_loss=1.0)


def test_nonfinite_outcomes_from_actual_zero_price_mark_are_unavailable_not_zero_returns():
    from ml4t.diagnostic.signal import prepare_signal_panel

    factor, forward, dates, _ = _inputs(zero_mark=True)
    assert np.isinf(factor.loc[(dates[3], "C")])
    assert np.isinf(forward.loc[(dates[2], "C"), "1D"])
    panel = prepare_signal_panel(factor, forward, quantiles=2, max_loss=0.6)
    assert panel.data.height == 10
    assert panel.metadata["loss_by_stage"] == {
        "factor": 5,
        "returns": 9,
        "groups": 0,
        "bucketing": 0,
    }
    keys = {
        (pd.Timestamp(day).value, asset)
        for day, asset in panel.data.select("date", "asset").iter_rows()
    }
    assert (dates[2].value, "C") not in keys and (dates[3].value, "C") not in keys


def test_printing_prepared_panel_does_not_dump_the_entire_caller_arrow_payload():
    from ml4t.diagnostic.signal import prepare_signal_panel

    factor, forward, _, _ = _inputs()
    text = repr(prepare_signal_panel(factor, forward, quantiles=2, max_loss=0.5))
    assert text.startswith("PreparedSignalPanel(")
    assert len(text) < 200


def test_float32_feed_values_use_float64_analysis_not_float32_accumulation():
    from ml4t.diagnostic.signal import prepare_signal_panel

    factor, forward, dates, _ = _inputs()
    factor, forward = factor.astype("float32"), forward.astype("float32")
    panel = prepare_signal_panel(factor, forward, quantiles=2, max_loss=0.5)
    result = panel.analyze(min_assets=2)
    for column in ["1D", "2D"]:
        values_by_bucket = {1: [], 2: []}
        for day in dates[1:4]:
            for position, asset in enumerate(factor.loc[day].sort_values().index):
                values_by_bucket[1 if position < 2 else 2].append(
                    float(forward.loc[(day, asset), column])
                )
        for bucket, values in values_by_bucket.items():
            mean = math.fsum(values) / len(values)
            std = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1))
            assert result.quantile_returns[column][bucket] == pytest.approx(
                mean, rel=0.0, abs=1e-15
            )
            assert result.quantile_returns_std[column][bucket] == pytest.approx(
                std, rel=0.0, abs=1e-15
            )


def test_requested_return_columns_determine_the_common_sample_not_unused_tails():
    from ml4t.diagnostic.signal import prepare_signal_panel

    factor, forward, dates, _ = _inputs()
    one = prepare_signal_panel(factor, forward, return_columns=["1D"], quantiles=2, max_loss=1 / 3)
    both = prepare_signal_panel(
        factor, forward, return_columns=["2D", "1D"], quantiles=2, max_loss=0.5
    )
    assert one.data.height == 16 and both.data.height == 12
    assert one.metadata["loss_by_stage"] == {"factor": 4, "returns": 4, "groups": 0, "bucketing": 0}
    assert one.data["date"].n_unique() == 4
    assert both.analyze(min_assets=2).periods == (2, 1)
    assert pd.Timestamp(one.data["date"].max()) == dates[4]


@pytest.mark.parametrize(
    "producer",
    [
        "normal",
        "zero_mark",
        "group_gap",
        "partial",
        "constant",
        "nanosecond",
        "custom_keys",
        "date_keys",
    ],
)
def test_excluded_observation_keys_and_reasons_are_owned_exact_and_survive_analysis_json(
    producer, tmp_path
):
    from ml4t.diagnostic import SignalResult
    from ml4t.diagnostic.signal import prepare_signal_panel

    factor, forward, dates, assets = _inputs(
        nanosecond=producer == "nanosecond", zero_mark=producer == "zero_mark"
    )
    expected = {(day.value, asset): "returns" for day in dates[4:] for asset in assets}
    expected.update({(dates[0].value, asset): "factor" for asset in assets})
    options = {"quantiles": 2}
    if producer == "zero_mark":
        expected[(dates[3].value, "C")] = "factor"
        expected[(dates[2].value, "C")] = "returns"
    elif producer == "group_gap":
        groups = (
            pd.Series(
                ["g1" if asset in ["A", "B"] else "g2" for _ in dates for asset in assets],
                index=factor.index,
            )
            .drop((dates[2], "A"))
            .reindex(factor.index)
        )
        options.update(groups=groups, binning_by_group=True)
        expected.update({(dates[2].value, "A"): "groups", (dates[2].value, "B"): "bucketing"})
    elif producer == "partial":
        factor = factor - factor.groupby(level="date").transform("mean")
        options["quantiles"] = [0.25, 0.75]
        for day in dates[1:4]:
            for asset in factor.loc[day].sort_values().iloc[[0, -1]].index:
                expected[(day.value, asset)] = "bucketing"
    elif producer == "constant":
        factor.loc[dates[2]] = factor.loc[dates[2]].mean()
        expected.update({(dates[2].value, asset): "bucketing" for asset in assets})
    date_key, asset_key = "date", "asset"
    if producer == "custom_keys":
        date_key, asset_key = "timestamp", "symbol"
        factor.index = factor.index.set_names([date_key, asset_key])
        forward.index = forward.index.set_names([date_key, asset_key])
        options.update(date_col=date_key, asset_col=asset_key)
    elif producer == "date_keys":
        factor = pl.from_pandas(factor.reset_index()).with_columns(pl.col("date").cast(pl.Date))
        forward = pl.from_pandas(forward.reset_index()).with_columns(pl.col("date").cast(pl.Date))
    panel = prepare_signal_panel(factor, forward, max_loss=1.0, **options)
    excluded = panel.excluded_data
    actual = dict(
        zip(
            zip(
                excluded[date_key].dt.timestamp("ns").to_list(),
                excluded[asset_key].to_list(),
                strict=True,
            ),
            excluded["exclusion_stage"].to_list(),
            strict=True,
        )
    )
    assert actual == expected and excluded.height == len(expected)
    assert excluded["exclusion_reason"].null_count() == 0
    for stage, reason in excluded.select("exclusion_stage", "exclusion_reason").iter_rows():
        expected_reason = {
            "factor": "missing or nonfinite factor",
            "returns": "missing or nonfinite requested outcome",
            "groups": "missing or nonfinite group",
        }.get(stage)
        if expected_reason is not None:
            assert reason == expected_reason
        elif producer == "partial":
            assert reason == "outside requested bucket edges"
        else:
            assert "Bin edges must be unique" in reason
    if producer == "nanosecond":
        assert excluded["date"].dtype == pl.Datetime("ns", "UTC")
    retained = set(zip(panel.data[date_key].dt.timestamp("ns"), panel.data[asset_key], strict=True))
    assert not retained.intersection(actual)
    assert len(retained) + len(actual) == len(factor)
    excluded.replace_column(
        excluded.get_column_index("exclusion_stage"),
        pl.Series("exclusion_stage", ["changed"] * excluded.height),
    )
    assert "changed" not in panel.excluded_data["exclusion_stage"].to_list()
    if producer == "normal":
        result = panel.analyze(min_assets=2)
        path = tmp_path / "with-exclusions.json"
        result.to_json(str(path))
        restored = SignalResult.from_json(str(path))
        encoded = restored.preparation_metadata["cleaning"]["excluded_data_ipc_base64"]
        assert pl.read_ipc(BytesIO(base64.b64decode(encoded))).equals(panel.excluded_data)


def test_already_complete_causal_sample_has_an_owned_empty_exclusion_schema():
    from ml4t.diagnostic.signal import prepare_signal_panel

    factor, forward, dates, _ = _inputs()
    factor = factor.loc[(slice(dates[1], dates[3]), slice(None))]
    panel = prepare_signal_panel(factor, forward, quantiles=2, max_loss=0.0)
    assert panel.data.height == 12
    excluded = panel.excluded_data
    assert excluded.height == 0
    assert excluded.columns == ["date", "asset", "exclusion_stage", "exclusion_reason"]
    assert excluded["date"].dtype == panel.data["date"].dtype


@pytest.mark.parametrize(
    "options",
    [
        {"quantiles": None, "bins": None},
        {"quantiles": 2, "bins": 2},
        {"quantiles": [0.0, 0.5, 0.5, 1.0]},
        {"quantiles": [-0.1, 1.0]},
        {"quantiles": [0.0, 1.1]},
        {"quantiles": None, "bins": [0.0, 1.0, 1.0]},
        {"zero_aware": "yes"},
        {"quantiles": [0.0, 0.5, 1.0], "zero_aware": True},
        {"binning_by_group": "no"},
        {"quantiles": np.array([[0.0, 0.5], [0.6, 1.0]])},
        {"max_loss": np.nan},
        {"max_loss": np.inf},
        {"max_loss": True},
        {"max_loss": -0.1},
        {"max_loss": 1.1},
        {"return_columns": []},
        {"return_columns": ["1D", "1D"]},
    ],
)
def test_invalid_policy_is_rejected_not_suppressed_as_a_total_bucket_loss(options):
    from ml4t.diagnostic.signal import prepare_signal_panel

    factor, forward, _, _ = _inputs()
    kwargs = {"quantiles": 2, "max_loss": 1.0}
    kwargs.update(options)
    if "max_loss" in options:
        message = "max_loss"
    elif "return_columns" in options:
        message = "return_columns"
    elif "binning_by_group" in options:
        message = "binning_by_group"
    elif "zero_aware" in options:
        message = "zero_aware"
    elif "bins" in options and options["bins"] is not None and options["quantiles"] is None:
        message = "bins.*edges"
    elif options.get("quantiles", 2) is None or options.get("bins") is not None:
        message = "exactly one"
    else:
        message = "edges|probabilities"
    groups = {"A": "g1", "B": "g1", "C": "g2", "D": "g2"}
    with pytest.raises(ValueError, match=message):
        prepare_signal_panel(factor, forward, groups=groups, **kwargs)
