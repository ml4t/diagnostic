"""Portfolio NAV contributions require a supplied denominator and declared clock."""

import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.evaluation import PortfolioAnalysis


def _account(*, calendar=None, feed_spec=None, fills=None):
    dates = pd.date_range("2025-07-07", periods=3, tz="UTC")
    if fills is None:
        fills = pd.DataFrame(
            {
                "timestamp": dates,
                "asset": [7] * 3,
                "quantity": [10.0, 5.0, -12.0],
                "price": [100.0, 110.0, 120.0],
                "commission": [10.0, 10.0, 12.0],
            }
        )
    return PortfolioAnalysis.from_frames(
        pd.Series([0.01] * 3, index=dates),
        transactions=fills,
        calendar=calendar,
        feed_spec=feed_spec,
    )


def _nav(value=1000.0):
    return pd.Series([value], index=pd.DatetimeIndex(["2025-07-09T00:00Z"]))


def test_supplied_beginning_nav_gives_portfolio_contribution_without_aliasing_lot_return():
    result = _account().compute_round_trips(initial_state="flat")
    nav = _nav()
    before = nav.copy(deep=True)
    contributions = result.portfolio_contributions(nav)
    assert contributions["gross_nav_contribution"].to_list() == [0.22]
    assert contributions["net_nav_contribution"].to_list() == [0.194]
    assert contributions["entry_notional"].to_list() == [1220.0]
    assert contributions["gross_lot_return"].to_list() == [pytest.approx(220 / 1220)]
    assert contributions["portfolio_nav"].to_list() == [1000.0]
    assert contributions["calendar"].to_list() == [None]
    pd.testing.assert_series_equal(nav, before)


@pytest.mark.parametrize("value", [0.0, -10.0, None, float("nan")])
def test_missing_or_nonpositive_nav_cannot_produce_a_finite_portfolio_return(value):
    result = _account().compute_round_trips(initial_state="flat")
    contributions = result.portfolio_contributions(_nav(value))
    assert contributions["gross_pnl"].to_list() == [220.0]
    assert contributions["gross_nav_contribution"].to_list() == [None]
    assert contributions["normalization_status"].to_list() == ["unavailable_nav"]


def test_missing_nav_row_is_not_forward_filled_from_a_prior_account_observation():
    result = _account().compute_round_trips(initial_state="flat")
    prior = _nav()
    prior.index = pd.DatetimeIndex(["2025-07-08T00:00Z"])
    contributions = result.portfolio_contributions(prior)
    assert contributions["portfolio_nav"].to_list() == [None]
    assert contributions["net_nav_contribution"].to_list() == [None]


def test_calendar_none_row_mode_does_not_manufacture_date_labels_into_instants():
    result = _account().compute_round_trips(initial_state="flat")
    dates = pl.DataFrame({"date": [pd.Timestamp("2025-07-09").date()], "equity": [1000.0]})
    with pytest.raises(ValueError, match="keys must match"):
        result.portfolio_contributions(dates)
    civil = result.portfolio_contributions(dates, calendar=None, alignment="civil_date")
    assert civil["net_nav_contribution"].to_list() == [0.194]
    assert civil["nav_alignment"].to_list() == ["civil_date"]


def test_calendar_precedence_uses_owned_explicit_none_before_declared_nyse():
    analysis = _account(calendar=None, feed_spec={"time_contract_version": 2, "calendar": "NYSE"})
    result = analysis.compute_round_trips(initial_state="flat")
    assert result.portfolio_contributions(_nav())["calendar"].to_list() == [None]
    # An explicit method argument overrides the preserved construction argument.
    values = pl.DataFrame({"date": [pd.Timestamp("2025-07-09").date()], "equity": [1000.0]})
    with pytest.raises(ValueError, match="outside"):
        result.portfolio_contributions(values, calendar="NYSE")


def test_nyse_and_overnight_cme_sessions_use_existing_authoritative_session_mapping():
    dates = pd.to_datetime(["2025-07-07T15:00Z", "2025-07-08T15:00Z"])
    fills = pd.DataFrame(
        {
            "timestamp": dates,
            "asset": [7, 7],
            "quantity": [10.0, -10.0],
            "price": [100.0, 120.0],
            "commission": [1.0, 1.0],
        }
    )
    result = _account(calendar="NYSE", fills=fills).compute_round_trips(initial_state="flat")
    nav = pl.DataFrame({"date": [pd.Timestamp("2025-07-08").date()], "equity": [1000.0]})
    assert result.portfolio_contributions(nav)["net_nav_contribution"].to_list() == [0.198]
    fills["timestamp"] = pd.to_datetime(["2025-07-06T23:30Z", "2025-07-07T00:30Z"])
    overnight = _account(calendar="CME_Equity", fills=fills).compute_round_trips(
        initial_state="flat"
    )
    nav = pl.DataFrame({"date": [pd.Timestamp("2025-07-07").date()], "equity": [1000.0]})
    assert overnight.portfolio_contributions(nav)["net_nav_contribution"].to_list() == [0.198]


def test_same_instants_in_other_known_timezones_match_without_losing_ns():
    result = _account().compute_round_trips(initial_state="flat")
    nav = _nav()
    nav.index = nav.index.tz_convert("America/New_York")
    assert result.portfolio_contributions(nav)["net_nav_contribution"].to_list() == [0.194]
    nav.index = nav.index.tz_localize(None)
    with pytest.raises(ValueError, match="keys must match"):
        result.portfolio_contributions(nav)


def test_duplicate_session_nav_is_ambiguous_and_does_not_multiply_closing_trade_rows():
    dates = pd.to_datetime(["2025-07-07T15:00Z", "2025-07-08T15:00Z"])
    fills = pd.DataFrame(
        {
            "timestamp": dates,
            "asset": [7, 7],
            "quantity": [10.0, -10.0],
            "price": [100.0, 120.0],
            "commission": [1.0, 1.0],
        }
    )
    result = _account(calendar="NYSE", fills=fills).compute_round_trips(initial_state="flat")
    nav = pd.Series(
        [1000.0, 1100.0], index=pd.to_datetime(["2025-07-08T15:00Z", "2025-07-08T16:00Z"])
    )
    before = nav.copy(deep=True)
    with pytest.raises(ValueError, match="duplicate"):
        result.portfolio_contributions(nav)
    pd.testing.assert_series_equal(nav, before)


def test_explicit_nav_column_aliases_and_unknown_fees_preserve_gross_and_net_missingness():
    dates = pd.date_range("2025-07-07", periods=2, tz="UTC")
    fills = pd.DataFrame(
        {"timestamp": dates, "asset": [7, 7], "quantity": [10.0, -10.0], "price": [100.0, 120.0]}
    )
    result = _account(fills=fills).compute_round_trips(initial_state="flat")
    nav = pd.DataFrame({"when": dates[-1:], "nav": [1000.0]})
    contributions = result.portfolio_contributions(nav, timestamp_col="when", value_col="nav")
    assert contributions["gross_nav_contribution"].to_list() == [0.2]
    assert contributions["net_nav_contribution"].to_list() == [None]


def test_empty_actual_closing_sample_retains_a_typed_empty_contribution_table():
    fills = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-07-07T00:00Z"]),
            "asset": [7],
            "quantity": [1.0],
            "price": [100.0],
            "commission": [0.0],
        }
    )
    result = _account(fills=fills).compute_round_trips(initial_state="flat")
    assert result.portfolio_contributions(_nav()).is_empty()


def test_real_published_pyfolio_nav_contribution_agrees_on_declared_civil_date_profile():
    reference = pytest.importorskip("pyfolio.round_trips")
    result = _account().compute_round_trips(initial_state="flat")
    fills = (
        result.executions.to_pandas()
        .set_index("timestamp")
        .rename(columns={"asset": "symbol", "quantity": "amount"})
    )
    fills["symbol"] = fills["symbol"].map({7: "A"})
    values = _nav()
    legacy = reference.extract_round_trips(fills, portfolio_value=values)
    native = result.portfolio_contributions(values, calendar=None, alignment="civil_date")
    assert native["gross_nav_contribution"].to_list() == legacy["returns"].to_list() == [0.22]
    # Published reference does not subtract the supplied commissions here.
    assert native["net_nav_contribution"].to_list() == [0.194]


def test_finite_but_unrepresentable_nav_ratio_raises_instead_of_returning_infinity():
    result = _account().compute_round_trips(initial_state="flat")
    with pytest.raises(ValueError, match="finite.*contribution|contribution.*finite"):
        result.portfolio_contributions(_nav(1e-308))
