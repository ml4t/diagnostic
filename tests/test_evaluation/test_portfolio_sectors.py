"""Sector labels must not erase real holdings, change their clock or overwrite cash."""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from test_portfolio_positions import _actual_prepared_inputs

from ml4t.diagnostic.evaluation import PortfolioAnalysis


def _analysis():
    returns, positions, valuations, account = _actual_prepared_inputs()
    return PortfolioAnalysis(returns, positions=positions), returns, positions, valuations, account


def _assert_conservation(result):
    grouped = (
        result.sectors.group_by("date")
        .agg(pl.col("gross_exposure").sum(), pl.col("net_exposure").sum())
        .sort("date")
    )
    np.testing.assert_allclose(grouped["gross_exposure"], [0, 700, 700])
    np.testing.assert_allclose(grouped["net_exposure"], [0, 100, 100])
    assert grouped["date"].to_list() == result.timeline["date"].to_list()


@pytest.mark.parametrize("shape", ["dict", "series", "pandas", "polars"])
def test_actual_sector_values_and_ratios_reconcile_for_all_static_mapping_routes(shape):
    analysis, returns, _, valuations, account = _analysis()
    mapping = {"A": "Technology", "B": "Finance"}
    if shape == "series":
        mapping = pd.Series(mapping)
    elif shape in {"pandas", "polars"}:
        mapping = pd.DataFrame({"asset": ["A", "B"], "sector": ["Technology", "Finance"]})
        if shape == "polars":
            mapping = pl.from_pandas(mapping)
    result = analysis.compute_holdings_analysis(
        valuations, complete_snapshots=True, sector_mapping=mapping
    )
    _assert_conservation(result)
    last = result.sectors.filter(pl.col("date") == returns.index[-1])
    technology = last.filter(pl.col("sector") == "Technology").row(0, named=True)
    finance = last.filter(pl.col("sector") == "Finance").row(0, named=True)
    assert technology["long_exposure"] == technology["net_exposure"] == 400
    assert finance["short_exposure"] == finance["gross_exposure"] == 300
    assert finance["net_exposure"] == -300
    assert technology["open_positions"] == finance["open_positions"] == 1
    assert technology["net_exposure_fraction"] == pytest.approx(0.4)
    assert finance["net_exposure_fraction"] == pytest.approx(-0.3)
    assert finance["gross_exposure_fraction"] == pytest.approx(0.3)
    assert result.timeline["cash"].to_list() == account["cash"].to_list()


def test_same_named_cash_sector_cannot_overwrite_account_cash_or_net_holdings():
    analysis, returns, _, valuations, _ = _analysis()
    result = analysis.compute_holdings_analysis(
        valuations, complete_snapshots=True, sector_mapping={"A": "cash", "B": "cash"}
    )
    _assert_conservation(result)
    last = result.sectors.filter(pl.col("date") == returns.index[-1]).row(0, named=True)
    assert last["sector"] == "cash" and last["net_exposure"] == 100
    assert last["gross_exposure"] == 700 and last["open_positions"] == 2
    assert result.timeline["cash"][-1] == 900
    assert result.timeline["cash_fraction"][-1] == pytest.approx(0.9)


@pytest.mark.parametrize("mapping", [{"A": "Technology"}, {"A": "Unknown", "B": None}, {}])
def test_unknown_sector_group_retains_unmapped_actual_short_instead_of_dropping_it(mapping):
    analysis, returns, _, valuations, _ = _analysis()
    result = analysis.compute_holdings_analysis(
        valuations, complete_snapshots=True, sector_mapping=mapping
    )
    _assert_conservation(result)
    last = result.sectors.filter(pl.col("date") == returns.index[-1])
    unknown = last.filter(pl.col("sector").is_null()).row(0, named=True)
    assert unknown["net_exposure"] == (-300 if mapping else 100)
    assert unknown["gross_exposure"] == (300 if mapping else 700)
    if mapping.get("A") == "Unknown":
        assert last.filter(pl.col("sector") == "Unknown")["net_exposure"].to_list() == [400]


@pytest.mark.parametrize("frame", ["pandas", "polars", "indexed_pandas"])
def test_dated_sector_changes_use_exact_observation_keys_without_filling_labels(frame):
    analysis, returns, _, valuations, _ = _analysis()
    dates = returns.index
    mapping = pd.DataFrame(
        {
            "date": [dates[-1], *dates],
            "asset": ["A", "B", "B", "B"],
            "sector": ["Technology", "Finance", "Finance", "Finance"],
        }
    )
    if frame == "polars":
        mapping = pl.from_pandas(mapping)
        mapping = mapping.with_columns(
            pl.col("date").cast(pl.Datetime("ms", mapping["date"].dtype.time_zone))
        )
    elif frame == "indexed_pandas":
        mapping = mapping.set_index("date")
    result = analysis.compute_holdings_analysis(
        valuations, complete_snapshots=True, sector_mapping=mapping
    )
    _assert_conservation(result)
    earlier = result.sectors.filter(pl.col("date") == dates[1])
    assert earlier.filter(pl.col("sector").is_null())["net_exposure"].to_list() == [400]
    assert earlier.filter(pl.col("sector") == "Technology").is_empty()
    later = result.sectors.filter(pl.col("date") == dates[-1])
    assert later.filter(pl.col("sector") == "Technology")["net_exposure"].to_list() == [400]
    assert result.positions["date"].to_list().count(dates[-1]) == 2


@pytest.mark.parametrize("equity", [None, 0.0, -1000.0])
def test_unusable_nav_keeps_raw_sector_values_and_unknown_classifications_without_zero_ratios(
    equity,
):
    analysis, _, _, valuations, _ = _analysis()
    valuations = valuations.with_columns(pl.lit(equity, dtype=pl.Float64).alias("equity"))
    result = analysis.compute_holdings_analysis(
        valuations, complete_snapshots=True, sector_mapping={"A": "Technology"}
    )
    _assert_conservation(result)
    for name in ["long", "short", "gross", "net"]:
        assert result.sectors[f"{name}_exposure_fraction"].null_count() == result.sectors.height
    assert result.timeline["cash"][-1] == 900


def test_cash_only_snapshots_have_typed_empty_sector_table_and_unchanged_cash():
    _, returns, _, valuations, _ = _analysis()
    positions = pd.DataFrame({"cash": [1000.0] * 3}, index=returns.index)
    valuations = valuations.with_columns(pl.lit(1000.0).alias("cash"))
    result = PortfolioAnalysis(returns, positions=positions).compute_holdings_analysis(
        valuations, complete_snapshots=True, sector_mapping={}
    )
    assert (
        result.sectors.is_empty()
        and result.sectors.schema["date"] == result.timeline.schema["date"]
    )
    assert result.timeline["gross_exposure_fraction"].to_list() == [0, 0, 0]
    assert result.timeline["cash"].to_list() == [1000, 1000, 1000]


def test_integer_asset_ids_numeric_sector_codes_and_missing_code_preserve_identity():
    _, returns, positions, valuations, _ = _analysis()
    long = positions.reset_index().melt(id_vars="date", var_name="asset", value_name="value")
    long["asset"] = long["asset"].map({"A": 1, "B": 2})
    analysis = PortfolioAnalysis(returns, positions=long)
    result = analysis.compute_holdings_analysis(
        valuations, complete_snapshots=True, sector_mapping=pd.Series([10.0, np.nan], index=[1, 2])
    )
    _assert_conservation(result)
    assert set(result.positions["asset"]) == {1, 2}
    assert result.sectors.filter(pl.col("sector") == 10)["net_exposure"][-1] == 400
    assert result.sectors.filter(pl.col("sector").is_null())["net_exposure"][-1] == -300


def test_sector_inputs_and_results_are_independent_owned_snapshots():
    analysis, _, _, valuations, _ = _analysis()
    mapping = pd.DataFrame({"asset": ["A", "B"], "sector": ["Technology", "Finance"]})
    before = mapping.copy(deep=True)
    result = analysis.compute_holdings_analysis(
        valuations, complete_snapshots=True, sector_mapping=mapping
    )
    pd.testing.assert_frame_equal(mapping, before)
    mapping.iloc[0, 1] = "Changed"
    assert "Changed" not in result.sectors["sector"].to_list()
    result.sectors.replace_column(1, pl.Series("sector", ["Corrupted"] * result.sectors.height))
    again = analysis.compute_holdings_analysis(
        valuations, complete_snapshots=True, sector_mapping=before
    )
    assert set(again.sectors["sector"]) == {"Technology", "Finance"}
    _assert_conservation(again)


@pytest.mark.parametrize(
    "defect", ["static_duplicate", "dated_duplicate", "timezone", "missing_date"]
)
def test_ambiguous_sector_keys_fail_without_expanding_or_mutating_holdings(defect):
    analysis, returns, positions, valuations, _ = _analysis()
    analysis.compute_holdings_analysis(valuations, complete_snapshots=True, sector_mapping={})
    if defect == "static_duplicate":
        mapping = pd.DataFrame({"asset": ["A", "A"], "sector": ["Technology", "Finance"]})
    else:
        dates = returns.index
        mapping = pd.DataFrame(
            {
                "date": [dates[-1], dates[-1]],
                "asset": ["A", "B"],
                "sector": ["Technology", "Finance"],
            }
        )
        if defect == "dated_duplicate":
            mapping.loc[1, "asset"] = "A"
        elif defect == "timezone":
            mapping["date"] = (
                mapping["date"].dt.tz_localize("UTC")
                if dates.tz is None
                else mapping["date"].dt.tz_localize(None)
            )
        else:
            mapping.loc[0, "date"] = pd.NaT
    original = mapping.copy(deep=True)
    with pytest.raises(ValueError, match="unique|timezone|nonmissing|date"):
        analysis.compute_holdings_analysis(
            valuations, complete_snapshots=True, sector_mapping=mapping
        )
    pd.testing.assert_frame_equal(mapping, original)
    assert positions.iloc[-1].to_list() == [400, -300]


@pytest.mark.parametrize(
    "defect",
    [
        "wrong_id_type",
        "missing_id",
        "empty_label",
        "infinite_label",
        "boolean_label",
        "extra_column",
    ],
)
def test_invalid_classification_metadata_fails_after_valid_public_workflow_control(defect):
    analysis, _, _, valuations, _ = _analysis()
    analysis.compute_holdings_analysis(valuations, complete_snapshots=True, sector_mapping={})
    mappings = {
        "wrong_id_type": {1: "Technology"},
        "missing_id": pd.DataFrame({"asset": [None], "sector": ["Technology"]}),
        "empty_label": {"A": ""},
        "infinite_label": {"A": float("inf")},
        "boolean_label": {"A": True},
        "extra_column": pd.DataFrame(
            {"asset": ["A"], "sector": ["Technology"], "known_at": ["future"]}
        ),
    }
    with pytest.raises((ValueError, TypeError), match="asset IDs|sector labels|mapping columns"):
        analysis.compute_holdings_analysis(
            valuations, complete_snapshots=True, sector_mapping=mappings[defect]
        )


def test_dated_mapping_cannot_round_nanoseconds_into_a_different_holdings_key():
    _, returns, positions, valuations, _ = _analysis()
    tz = valuations["date"].dtype.time_zone
    dates = pl.Series("date", returns.index).cast(pl.Datetime("us", tz))
    analysis = PortfolioAnalysis(returns, positions=positions, dates=dates)
    analysis.compute_holdings_analysis(valuations, complete_snapshots=True, sector_mapping={})
    mapping = pd.DataFrame(
        {
            "date": [returns.index[-1] + pd.Timedelta(1, unit="ns")],
            "asset": ["A"],
            "sector": ["Technology"],
        }
    )
    with pytest.raises(ValueError, match="precision"):
        analysis.compute_holdings_analysis(
            valuations, complete_snapshots=True, sector_mapping=mapping
        )


def test_unheld_classifications_do_not_expand_the_observed_holdings_sample():
    analysis, _, _, valuations, _ = _analysis()
    result = analysis.compute_holdings_analysis(
        valuations,
        complete_snapshots=True,
        sector_mapping={"A": "Technology", "B": "Finance", "C": "Unused"},
    )
    assert result.positions.height == 6 and "Unused" not in result.sectors["sector"].to_list()
    _assert_conservation(result)


def test_sector_requests_do_not_change_a_later_unclassified_result_or_cached_return_metrics():
    analysis, _, _, valuations, _ = _analysis()
    metrics = analysis.compute_summary_stats().to_dict()
    analysis.compute_holdings_analysis(
        valuations, complete_snapshots=True, sector_mapping={"A": "Technology", "B": "Finance"}
    )
    plain = analysis.compute_holdings_analysis(valuations, complete_snapshots=True)
    assert plain.sectors is None and "sector" not in plain.positions.columns
    assert plain.timeline["gross_exposure"].to_list() == [0, 700, 700]
    after = analysis.compute_summary_stats().to_dict()
    for name, value in metrics.items():
        if value is None:
            assert after[name] is None
        else:
            np.testing.assert_equal(after[name], value)
