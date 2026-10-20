"""Actual native outcome captures feed one cleaner without a producer dependency."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from ml4t.specs import FeedSpec

from ml4t.diagnostic.signal import SignalResult, prepare_signal_panel

DATA = Path(__file__).parent / "data/outcomes"


def _capture(missing=False, exit_bar=False):
    suffix = ("-exit-bar" if exit_bar else "") + ("-missing" if missing else "")
    return (
        pl.read_ipc(DATA / "factor.arrow"),
        pl.read_ipc(DATA / f"outcomes{suffix}.arrow"),
        json.loads((DATA / "provenance.json").read_text()),
    )


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("exit_bar", [False, True])
def test_actual_long_outcomes_preserve_endpoints_missingness_units_and_owned_json(
    missing, exit_bar, tmp_path
):
    factor, outcomes, provenance = _capture(missing, exit_bar)
    original = outcomes.clone()
    feed = FeedSpec.from_mapping(provenance["feed_spec"])
    panel = prepare_signal_panel(
        factor,
        outcomes,
        horizon_col="horizon",
        return_columns=["2ns", "1ns"],
        feed_spec=feed,
        quantiles=2,
        max_loss=1 / 6,
    )
    assert panel.metadata["return_columns"] == ["2ns", "1ns"]
    assert panel.metadata["loss_by_stage"]["returns"] == (2 if missing else 0)
    assert panel.data.height == (10 if missing else 12)
    assert panel.outcome_data.select(outcomes.columns).equals(outcomes)
    assert panel.outcome_data["window_unit"].unique().to_list() == ["elapsed"]
    assert panel.outcome_data["price_basis"].unique().to_list() == ["raw"]
    assert panel.outcome_data["endpoint_method"].unique().to_list() == [
        "exit_bar" if exit_bar else "cumulative"
    ]
    assert panel.outcome_data["duration_ns"].to_list() == [1, 2] * 12
    assert panel.outcome_data["return_duration_ns"].to_list() == (
        [1] * 24 if exit_bar else [1, 2] * 12
    )
    assert panel.metadata["calendar"] is None
    assert outcomes.equals(original)
    result = panel.analyze(min_assets=2)
    assert result.horizon_labels == ("2ns", "1ns")
    assert result.ic == {"2ns": pytest.approx(1.0), "1ns": pytest.approx(1.0)}
    assert result.outcome_data.equals(panel.outcome_data)
    path = tmp_path / "result.json"
    result.to_json(str(path))
    restored = SignalResult.from_json(str(path))
    assert restored.outcome_data.equals(panel.outcome_data)
    assert restored.outcome_data["timestamp"].dtype == pl.Datetime("ns", "UTC")
    owned = restored.outcome_data
    owned.replace_column(
        owned.get_column_index("forward_return"), pl.Series("forward_return", [9.0] * owned.height)
    )
    assert restored.outcome_data.equals(panel.outcome_data)
    prices = pl.read_ipc(DATA / "prices.arrow")
    for row in panel.data.with_columns(pl.col("timestamp").cast(pl.Int64)).iter_rows(named=True):
        for horizon, steps in (("1ns", 1), ("2ns", 2)):
            reference = row["timestamp"] + (steps - 1 if exit_bar else 0)
            start = prices.filter(
                (pl.col("asset") == row["asset"])
                & (pl.col("timestamp").cast(pl.Int64) == reference)
            ).item(0, "close")
            exit_price = prices.filter(
                (pl.col("asset") == row["asset"])
                & (pl.col("timestamp").cast(pl.Int64) == row["timestamp"] + steps)
            ).item(0, "close")
            assert row[horizon] == pytest.approx((exit_price - start) / start)


def test_long_producer_calendar_metadata_is_consumed_without_promoting_a_function_argument():
    factor, outcomes, _ = _capture()
    panel = prepare_signal_panel(
        factor,
        outcomes,
        horizon_col="horizon",
        timestamp_col="timestamp",
        entity_col="asset",
        quantiles=2,
        max_loss=0,
    )
    assert panel.metadata["calendar"] is None
    assert panel.metadata["calendar_source"] == "outcome_table"
    assert "calendar_argument" not in panel.metadata
    result = panel.analyze(min_assets=2)
    assert result.preparation_metadata["calendar"] is None
    assert result.preparation_metadata["calendar_source"] == "outcome_table"
    assert "calendar_argument" not in result.preparation_metadata


def test_overlapping_actual_outcome_batches_cannot_hide_duplicate_horizon_keys():
    factor, outcomes, provenance = _capture()
    duplicate = pl.concat([outcomes, outcomes])
    before = duplicate.clone()
    with pytest.raises(ValueError, match="duplicate"):
        prepare_signal_panel(
            factor, duplicate, horizon_col="horizon", feed_spec=provenance["feed_spec"]
        )
    assert duplicate.equals(before)


def test_owned_outcome_export_can_be_reingested_without_dropping_its_duration_fields():
    factor, outcomes, provenance = _capture()
    panel = prepare_signal_panel(
        factor,
        outcomes,
        horizon_col="horizon",
        feed_spec=provenance["feed_spec"],
        quantiles=2,
        max_loss=0,
    )
    exported = panel.analyze(min_assets=2).outcome_data
    restored = prepare_signal_panel(
        factor,
        exported,
        horizon_col="horizon",
        feed_spec=provenance["feed_spec"],
        quantiles=2,
        max_loss=0,
    )
    assert restored.outcome_data.equals(exported)
    assert restored.data.equals(panel.data)


def test_conflicting_real_producer_and_feed_calendars_reject_but_explicit_none_wins():
    factor, outcomes, _ = _capture()
    original = outcomes.clone()
    feed = FeedSpec(
        timestamp_col="timestamp", entity_col="asset", time_contract_version=2, calendar="NYSE"
    )
    with pytest.raises(ValueError, match="outcome calendar.*feed_spec"):
        prepare_signal_panel(factor, outcomes, horizon_col="horizon", feed_spec=feed)
    panel = prepare_signal_panel(
        factor,
        outcomes,
        horizon_col="horizon",
        feed_spec=feed,
        calendar=None,
        quantiles=2,
        max_loss=0,
    )
    assert panel.metadata["calendar"] is None
    assert panel.metadata["calendar_source"] == "argument"
    assert panel.analyze(min_assets=2).preparation_metadata["calendar"] is None
    assert outcomes.equals(original)


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("exit_bar", [False, True])
def test_optional_actual_pinned_producer_recreates_captured_samples(missing, exit_bar):
    labeling = pytest.importorskip("ml4t.engineer.labeling")
    from importlib.metadata import version

    factor, expected, provenance = _capture(missing, exit_bar)
    if version("ml4t-engineer") != provenance["producer_version"]:
        pytest.skip("this fixture pins its exact qualified native producer")
    actual = labeling.forward_returns(
        pl.read_ipc(DATA / ("prices-missing.arrow" if missing else "prices.arrow")).lazy(),
        horizons={"1ns": "1ns", "2ns": "2ns"},
        price_basis="raw",
        entries=factor.select("timestamp", "asset"),
        feed_spec=FeedSpec.from_mapping(provenance["feed_spec"]),
        window_unit="elapsed",
        available_at_col="known_at",
        missing_price_policy="null",
        tolerance="0ns",
        endpoint_method="exit_bar" if exit_bar else "cumulative",
    ).collect()
    assert actual.equals(expected)
    panel = prepare_signal_panel(
        factor,
        actual,
        horizon_col="horizon",
        feed_spec=provenance["feed_spec"],
        quantiles=2,
        max_loss=1 / 6,
    )
    np.testing.assert_allclose(list(panel.analyze(min_assets=2).ic.values()), [1.0, 1.0])
