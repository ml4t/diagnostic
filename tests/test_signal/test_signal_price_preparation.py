"""Price preparation feeds the same cleaner with explicit clocks and provenance."""

from fractions import Fraction

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.signal import prepare_signal_panel


@pytest.mark.parametrize("dtype,base", [(pl.Int64, 2**53), (pl.UInt64, 2**63)])
def test_public_forward_return_helper_does_not_round_integer_tick_changes_to_zero(dtype, base):
    from ml4t.diagnostic.metrics import compute_forward_returns

    prices = pl.DataFrame(
        {"date": [1751898600000000000 + offset for offset in range(3)], "asset": ["A"] * 3}
    ).with_columns(
        pl.col("date").cast(pl.Datetime("ns", "UTC")),
        pl.Series("close", [base, base + 1, base + 2], dtype=dtype),
    )
    output = compute_forward_returns(prices, periods=[1], group_col="asset")
    expected = [float(Fraction(1, base)), float(Fraction(1, base + 1))]
    np.testing.assert_allclose(output["fwd_ret_1"][:2], expected, rtol=3e-16, atol=0)
    assert output["fwd_ret_1"][-1] is None


def test_exactly_representable_float32_quotes_use_float64_return_arithmetic():
    from ml4t.diagnostic.metrics import compute_forward_returns

    base = 2**24
    prices = pl.DataFrame(
        {"date": [1751898600000000000 + offset for offset in range(3)], "asset": ["A"] * 3}
    ).with_columns(
        pl.col("date").cast(pl.Datetime("ns", "UTC")),
        pl.Series("close", [base, base + 2, base + 4], dtype=pl.Float32),
    )
    assert prices["close"].to_list() == [base, base + 2, base + 4]
    result = compute_forward_returns(prices, periods=[1], group_col="asset")
    np.testing.assert_allclose(
        result["fwd_ret_1"][:2],
        [float(Fraction(2, base)), float(Fraction(2, base + 2))],
        rtol=3e-16,
        atol=0,
    )
    assert result["fwd_ret_1"].dtype == pl.Float64


def _holiday_inputs(*, missing_row=False):
    dates = pd.to_datetime(
        ["2025-07-02T15:00Z", "2025-07-03T15:00Z", "2025-07-07T15:00Z", "2025-07-08T15:00Z"]
    )
    factor = pd.Series(
        [-1.0, 1.0] * 4,
        index=pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "asset"]),
        name="factor",
    )
    prices = pl.DataFrame(
        {
            "date": np.repeat(dates.as_unit("ns").asi8, 2),
            "asset": ["A", "B"] * 4,
            "close": [100.0, 50.0, 101.0, 52.0, 98.0, None, 103.0, 53.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ns", "UTC")))
    if missing_row:
        prices = prices.filter(~((pl.col("asset") == "B") & (pl.col("date") == dates[2])))
    return factor, prices, dates


@pytest.mark.parametrize("missing_row", [False, True])
@pytest.mark.parametrize("index_mode", ["common_grid", "per_asset"])
def test_holiday_price_grid_does_not_silently_compress_absent_asset_rows(missing_row, index_mode):
    factor, prices, dates = _holiday_inputs(missing_row=missing_row)
    original = prices.clone()
    panel = prepare_signal_panel(
        factor,
        prices=prices,
        horizons={"one_obs": 1},
        price_basis="raw",
        calendar=None,
        index_mode=index_mode,
        quantiles=None,
        bins=[-2, 0, 2],
        max_loss=0.5,
    )
    data = panel.data
    assert data.height == (5 if missing_row and index_mode == "per_asset" else 4)
    assert prices.equals(original)
    assert panel.metadata["loss_by_stage"]["returns"] == 8 - data.height
    row = panel.outcome_data.filter((pl.col("date") == dates[1]) & (pl.col("asset") == "B"))
    if missing_row and index_mode == "per_asset":
        assert row["exit_time"][0] == dates[3]
        assert row["forward_return"][0] == pytest.approx(1 / 52)
    else:
        assert row["forward_return"][0] is None
        assert row["requested_exit_time"][0] == dates[2]
        assert row["exit_row_present"][0] == (not missing_row)
        assert row["exit_price_observed"][0] is False
    assert panel.outcome_data["window_unit"].unique().to_list() == ["observation"]
    assert panel.analyze(min_assets=2).outcome_data.equals(panel.outcome_data)


@pytest.mark.parametrize("missing_row", [False, True])
def test_forward_fill_is_an_explicit_owned_analytical_view_not_a_new_observed_quote(missing_row):
    factor, prices, dates = _holiday_inputs(missing_row=missing_row)
    original = prices.clone()
    panel = prepare_signal_panel(
        factor,
        prices=prices,
        horizons={"one_obs": 1},
        price_basis="raw",
        calendar=None,
        index_mode="common_grid",
        fill_method="ffill",
        quantiles=None,
        bins=[-2, 0, 2],
        max_loss=0.25,
    )
    assert panel.data.height == 6
    row = panel.outcome_data.filter((pl.col("date") == dates[1]) & (pl.col("asset") == "B"))
    assert row["forward_return"][0] == 0
    assert row["analytical_fill_used"][0] is True
    assert row["exit_price_observed"][0] is False
    assert row["exit_row_present"][0] == (not missing_row)
    assert prices.equals(original)


@pytest.mark.parametrize(
    "endpoint_method,expected", [("cumulative", [0.21, 0.44]), ("exit_bar", [0.1, 0.2])]
)
def test_cumulative_and_terminal_bar_outcomes_have_distinct_values_and_reference_keys(
    endpoint_method, expected
):
    dates = pd.date_range("2025-07-07T14:30Z", periods=3, freq="1h")
    prices = pd.DataFrame({"A": [100, 110, 121], "B": [100, 120, 144]}, index=dates)
    factor = pd.Series(
        [-1.0, 1.0],
        index=pd.MultiIndex.from_product([dates[:1], ["A", "B"]], names=["date", "asset"]),
        name="factor",
    )
    panel = prepare_signal_panel(
        factor,
        prices=prices,
        horizons={"two": 2},
        price_basis="raw",
        endpoint_method=endpoint_method,
        calendar=None,
        quantiles=2,
        max_loss=0,
    )
    np.testing.assert_allclose(panel.data.sort("asset")["two"], expected, rtol=1e-12, atol=1e-12)
    assert panel.outcome_data["exit_time"].to_list() == [dates[2]] * 2
    assert (
        panel.outcome_data["reference_time"].to_list()
        == [dates[0 if endpoint_method == "cumulative" else 1]] * 2
    )


def test_two_price_sources_are_not_silently_chosen_and_invalid_calls_leave_inputs_unchanged():
    factor, prices, _ = _holiday_inputs()
    original = prices.clone()
    with pytest.raises(ValueError, match="prices.*forward_returns|forward_returns.*prices"):
        prepare_signal_panel(factor, prices, prices=prices, price_basis="raw")
    with pytest.raises(ValueError, match="price_basis"):
        prepare_signal_panel(factor, prices=prices, horizons={"one": 1})
    assert prices.equals(original)


def test_return_outlier_filter_uses_all_factor_dates_before_sparse_factor_row_selection():
    dates = pd.date_range("2025-07-07T14:30Z", periods=4, freq="1h")
    prices = pd.DataFrame({"A": [100, 110, 220, 242], "B": [100] * 4}, index=dates)
    index = pd.MultiIndex.from_product([dates[:3], ["A", "B"]], names=["date", "asset"])
    factor = pd.Series([-1.0, 1.0] * 3, index=index, name="factor").drop((dates[1], "A"))
    panel = prepare_signal_panel(
        factor,
        prices=prices,
        horizons={"one": 1},
        price_basis="raw",
        calendar=None,
        return_filter_zscore=0.5,
        quantiles=None,
        bins=[-2, 0, 2],
        max_loss=0.4,
    )
    assert panel.data["asset"].to_list() == ["B"] * 3
    assert panel.outcome_data.filter(pl.col("outcome_status") == "filtered_outlier")[
        "asset"
    ].to_list() == ["A", "A"]
    assert panel.metadata["loss_by_stage"]["returns"] == 2


def test_valid_extreme_price_path_does_not_overflow_the_return_filter_dispersion():
    dates = pd.date_range("2025-07-07T14:30Z", periods=4, freq="1h")
    prices = pd.DataFrame({"A": [1e-308, 1.0, 1e308, 1.0], "B": [1.0] * 4}, index=dates)
    factor = pd.Series(
        [-1.0, 1.0] * 3,
        index=pd.MultiIndex.from_product([dates[:3], ["A", "B"]], names=["date", "asset"]),
        name="factor",
    )
    panel = prepare_signal_panel(
        factor,
        prices=prices,
        horizons={"one": 1},
        price_basis="raw",
        calendar=None,
        return_filter_zscore=1.0,
        quantiles=None,
        bins=[-2, 0, 2],
        max_loss=1 / 6,
    )
    rejected = panel.outcome_data.filter(pl.col("outcome_status") == "filtered_outlier")
    assert rejected["date"].to_list() == [dates[2]]
    assert rejected["asset"].to_list() == ["A"]
    assert panel.data.height == 5


def test_all_unknown_quote_values_are_audited_losses_not_a_missing_numeric_schema():
    factor, prices, _ = _holiday_inputs()
    unknown = prices.with_columns(pl.lit(None).alias("close"))
    before = unknown.clone()
    panel = prepare_signal_panel(
        factor,
        prices=unknown,
        horizons={"one": 1},
        price_basis="raw",
        calendar=None,
        quantiles=None,
        bins=[-2, 0, 2],
        max_loss=1,
    )
    assert panel.data.height == 0
    assert panel.metadata["loss_by_stage"]["returns"] == 8
    assert unknown.equals(before)


@pytest.mark.parametrize("identifiers", [[1001, 1002], pd.CategoricalIndex(["A", "B"])])
def test_wide_prices_preserve_numeric_identity_and_match_string_categorical_keys(identifiers):
    dates = pd.date_range("2025-07-07T14:30Z", periods=3, freq="1h")
    prices = pd.DataFrame(
        {identifiers[0]: [100, 110, 121], identifiers[1]: [100, 120, 144]}, index=dates
    )
    factor = pd.Series(
        [-1.0, 1.0],
        index=pd.MultiIndex.from_product([dates[:1], identifiers], names=["date", "asset"]),
        name="factor",
    )
    panel = prepare_signal_panel(
        factor,
        prices=prices,
        horizons={"two": 2},
        price_basis="raw",
        calendar=None,
        quantiles=2,
        max_loss=0,
    )
    assert set(panel.data["asset"]) == set(identifiers)
    assert panel.data["asset"].dtype == (pl.Int64 if isinstance(identifiers, list) else pl.String)
    np.testing.assert_allclose(
        panel.data.sort("asset")["two"], [0.21, 0.44], rtol=1e-12, atol=1e-12
    )


def _available_quotes(*, missing=False):
    dates = pd.date_range("2025-07-07T14:30Z", periods=3, freq="1h")
    ticks = np.repeat(dates.as_unit("ns").asi8, 2)
    prices = pl.DataFrame(
        {
            "date": ticks,
            "asset": ["A", "B"] * 3,
            "close": [100.0, 100.0, None if missing else 110.0, 120.0, 121.0, 144.0],
            "known_at": ticks + 30 * 10**9,
        }
    ).with_columns(pl.col("date", "known_at").cast(pl.Datetime("ns", "UTC")))
    factor = pd.Series(
        [-1.0, 1.0],
        index=pd.MultiIndex.from_product([dates[:1], ["A", "B"]], names=["date", "asset"]),
        name="factor",
    )
    return factor, prices, dates


@pytest.mark.parametrize("naive", [False, True])
def test_observed_price_availability_uses_declared_timezone_and_keeps_unknown_legs_unknown(naive):
    from ml4t.specs import FeedSpec

    factor, prices, dates = _available_quotes()
    prices = prices.with_columns(
        pl.when((pl.col("asset") == "A") & (pl.col("date") == dates[0]))
        .then(None)
        .otherwise(pl.col("known_at"))
        .alias("known_at")
    )
    if naive:
        prices = prices.with_columns(
            pl.col("known_at").dt.convert_time_zone("America/New_York").dt.replace_time_zone(None)
        )
    before = prices.clone()
    panel = prepare_signal_panel(
        factor,
        prices=prices,
        horizons={"two": 2},
        price_basis="raw",
        available_at_col="known_at",
        feed_spec=FeedSpec(
            timestamp_col="date",
            entity_col="asset",
            timezone="America/New_York",
            time_contract_version=2,
            calendar=None,
        ),
        quantiles=2,
        max_loss=0,
    )
    outcomes = panel.outcome_data.sort("asset")
    assert outcomes["outcome_available_at"].to_list() == [None, dates[2] + pd.Timedelta(seconds=30)]
    assert prices.equals(before)


def test_forward_filled_view_does_not_claim_the_future_outcome_was_already_available():
    factor, prices, _ = _available_quotes(missing=True)
    panel = prepare_signal_panel(
        factor,
        prices=prices,
        horizons={"one": 1},
        price_basis="raw",
        calendar=None,
        available_at_col="known_at",
        fill_method="ffill",
        quantiles=2,
        max_loss=0,
    )
    row = panel.outcome_data.filter(pl.col("asset") == "A")
    assert row["analytical_fill_used"].item() is True
    assert row["forward_return"].item() == 0
    assert row["outcome_available_at"].item() is None


def test_sample_fitted_filter_does_not_claim_causal_label_availability():
    factor, prices, dates = _available_quotes()
    panel = prepare_signal_panel(
        factor,
        prices=prices,
        horizons={"one": 1},
        price_basis="raw",
        calendar=None,
        available_at_col="known_at",
        return_filter_zscore=20,
        quantiles=2,
        max_loss=0,
    )
    assert panel.data.height == 2
    assert (
        panel.outcome_data["exit_available_at"].to_list()
        == [dates[1] + pd.Timedelta(seconds=30)] * 2
    )
    assert panel.outcome_data["outcome_available_at"].null_count() == 2
